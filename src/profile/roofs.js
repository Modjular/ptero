// Empirical roofline for whatever GPU this page is running on.
//
// The report's Phase 0 normalises each kernel against its *attainable* roof, which needs
// peak FLOP/s and peak bandwidth. Those come off a spec sheet on NVIDIA hardware. Here
// they cannot: Apple publishes no FLOP/s or bandwidth figure for the M-series GPU that
// is meaningful for this workload, and WGSL f32 compute cannot reach M5's neural
// accelerators anyway — so even a correct vendor peak would be the wrong roof.
//
// So measure it, ERT-style. Three probes:
//
//   computeRoof     FMA-saturated, no memory traffic          -> GFLOP/s
//   bandwidthRoof   streaming triad past any cache            -> GB/s
//   intensitySweep  the curve between them                    -> the actual ridge point
//
// The third is the one that matters most, because it locates the ridge *empirically*
// rather than as a ratio of two separately-measured peaks. On unified memory with a
// large system-level cache, that ratio is not trustworthy.
//
// Everything here is self-contained: its own pipelines, its own buffers, no engine
// involvement. It is also the only part of Phase 0 whose numbers are a property of the
// machine rather than of Cellpose.

const WG = 256;

// Two things here are load-bearing against the compiler, and both were found by the
// measurement coming out wrong:
//
//   - Eight *independent* chains. One dependent chain measures FMA latency rather than
//     throughput and reads several times low.
//   - Exactly one FMA per chain per trip, with `k` and `c` read from the uniform. An
//     earlier version unrolled the body 4x with literal constants and reported 9.3
//     TFLOP/s — 3x the swept plateau — because `a*k+c` applied four times with known
//     constants folds to a single `a*k'+c'`. Runtime constants and no unrolling remove
//     the closed form. If this number ever drifts far above the sweep's plateau again,
//     suspect folding first.
const FMA_WGSL = /* wgsl */`
struct P { iters:u32, n:u32, _a:u32, _b:u32 };
struct Q { k:f32, c:f32, _a:f32, _b:f32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(2) var<uniform> q: Q;
@group(0) @binding(1) var<storage,read_write> o: array<f32>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  var a0 = f32(i) * 0.5; var a1 = a0 + 1.0; var a2 = a0 + 2.0; var a3 = a0 + 3.0;
  var a4 = a0 + 4.0; var a5 = a0 + 5.0; var a6 = a0 + 6.0; var a7 = a0 + 7.0;
  let k = q.k; let c = q.c;
  for (var t = 0u; t < p.iters; t = t + 1u) {
    a0 = a0 * k + c; a1 = a1 * k + c; a2 = a2 * k + c; a3 = a3 * k + c;
    a4 = a4 * k + c; a5 = a5 * k + c; a6 = a6 * k + c; a7 = a7 * k + c;
  }
  if (i < p.n) { o[i] = a0 + a1 + a2 + a3 + a4 + a5 + a6 + a7; }
}`;

const TRIAD_WGSL = /* wgsl */`
struct P { n:u32, _a:u32, _b:u32, _c:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       a: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read>       b: array<vec4<f32>>;
@group(0) @binding(3) var<storage,read_write> o: array<vec4<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  o[i] = a[i] + 2.0 * b[i];   // vec4 so each lane issues a wide load, not a scalar one
}`;

// One vec4 read + one vec4 write (32 bytes), with `iters` rounds of FMA on four
// independent vec4 chains (32 FLOP per round). So AI = iters exactly, and sweeping
// `iters` walks arithmetic intensity across the knee one FLOP/byte at a time.
//
// The four chains matter: an earlier version carried a single dependent chain and its
// plateau came in ~37% below the pure-FMA roof, because it was measuring FMA latency
// rather than throughput and therefore placed the ridge too low.
const SWEEP_WGSL = /* wgsl */`
struct P { n:u32, iters:u32, _a:u32, _b:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       a: array<vec4<f32>>;
@group(0) @binding(2) var<storage,read_write> o: array<vec4<f32>>;
@compute @workgroup_size(${WG})
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.n) { return; }
  let v = a[i];
  var v0 = v; var v1 = v + 1.0; var v2 = v + 2.0; var v3 = v + 3.0;
  let k = vec4<f32>(1.0000001); let c = vec4<f32>(0.0000001);
  for (var t = 0u; t < p.iters; t = t + 1u) {
    v0 = v0 * k + c; v1 = v1 * k + c; v2 = v2 * k + c; v3 = v3 * k + c;
  }
  o[i] = v0 + v1 + v2 + v3;
}`;

const RESOLVE_USAGE = GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC;
const READ_USAGE = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;

function pipelineFor(device, code) {
  return device.createComputePipeline({
    layout: "auto",
    compute: { module: device.createShaderModule({ code }), entryPoint: "main" },
  });
}

// Median GPU-side duration of a dispatch, in nanoseconds. Falls back to wall clock via
// onSubmittedWorkDone when timestamps are unavailable — coarser, but the roofs are big
// enough (milliseconds) that it still works.
async function timeDispatch(device, pipeline, bindGroup, wg, reps = 7) {
  const ts = !!device.features?.has?.("timestamp-query");
  const run = async () => {
    if (ts) {
      const qs = device.createQuerySet({ type: "timestamp", count: 2 });
      const res = device.createBuffer({ size: 16, usage: RESOLVE_USAGE });
      const rb = device.createBuffer({ size: 16, usage: READ_USAGE });
      const enc = device.createCommandEncoder({ label: "roof" });
      const p = enc.beginComputePass({
        timestampWrites: { querySet: qs, beginningOfPassWriteIndex: 0, endOfPassWriteIndex: 1 },
      });
      p.setPipeline(pipeline); p.setBindGroup(0, bindGroup); p.dispatchWorkgroups(wg); p.end();
      enc.resolveQuerySet(qs, 0, 2, res, 0);
      enc.copyBufferToBuffer(res, 0, rb, 0, 16);
      device.queue.submit([enc.finish()]);
      await rb.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(rb.getMappedRange().slice(0));
      rb.unmap(); rb.destroy(); res.destroy(); qs.destroy?.();
      return Number(t[1] - t[0]);
    }
    const t0 = performance.now();
    const enc = device.createCommandEncoder({ label: "roof" });
    const p = enc.beginComputePass();
    p.setPipeline(pipeline); p.setBindGroup(0, bindGroup); p.dispatchWorkgroups(wg); p.end();
    device.queue.submit([enc.finish()]);
    await device.queue.onSubmittedWorkDone();
    return (performance.now() - t0) * 1e6;
  };
  await run();                       // discard: first run pays pipeline warm-up
  const out = [];
  for (let i = 0; i < reps; i++) out.push(await run());
  out.sort((a, b) => a - b);
  return { medianNs: out[out.length >> 1], minNs: out[0], maxNs: out[out.length - 1] };
}

function uniformBuf(device, ints) {
  const b = device.createBuffer({
    size: Math.max(16, ints.length * 4),
    usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(b, 0, new Uint32Array(ints));
  return b;
}

/** Peak achievable f32 FLOP/s with no memory traffic at all. */
export async function computeRoof(device, { threads = 1 << 20, iters = 2000 } = {}) {
  const pipe = pipelineFor(device, FMA_WGSL);
  const out = device.createBuffer({
    size: threads * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uni = uniformBuf(device, [iters, threads, 0, 0]);
  const kc = device.createBuffer({
    size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  device.queue.writeBuffer(kc, 0, new Float32Array([1.0000001, 0.0000001, 0, 0]));
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: kc } },
    ],
  });
  const t = await timeDispatch(device, pipe, bg, threads / WG);
  out.destroy(); uni.destroy(); kc.destroy();
  const flops = threads * iters * 8 * 2;   // 8 chains, 2 FLOP per FMA
  return { gflops: flops / t.medianNs, ...t, threads, iters };
}

/** Peak achievable streaming bandwidth: two reads and one write per element. */
export async function bandwidthRoof(device, { bytesPerBuffer = 64 << 20 } = {}) {
  const cap = device.limits?.maxStorageBufferBindingSize ?? bytesPerBuffer;
  const size = Math.min(bytesPerBuffer, cap);
  const n = Math.floor(size / 16);                 // vec4 count
  const nAligned = Math.floor(n / WG) * WG;
  const pipe = pipelineFor(device, TRIAD_WGSL);
  const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const a = device.createBuffer({ size: nAligned * 16, usage: U });
  const b = device.createBuffer({ size: nAligned * 16, usage: U });
  const o = device.createBuffer({ size: nAligned * 16, usage: U | GPUBufferUsage.COPY_SRC });
  const uni = uniformBuf(device, [nAligned, 0, 0, 0]);
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: a } },
      { binding: 2, resource: { buffer: b } }, { binding: 3, resource: { buffer: o } },
    ],
  });
  const t = await timeDispatch(device, pipe, bg, nAligned / WG);
  a.destroy(); b.destroy(); o.destroy(); uni.destroy();
  const bytes = nAligned * 16 * 3;
  return { gbps: bytes / t.medianNs, ...t, bytes, workingSetMB: (nAligned * 16 * 3) / (1 << 20) };
}

/**
 * Trace the roofline curve by sweeping a kernel's arithmetic intensity across the knee.
 * Returns one point per intensity plus the empirical ridge — the lowest AI at which the
 * kernel reaches ~95% of the best GFLOP/s the sweep ever saw.
 */
export async function intensitySweep(device, {
  bytesPerBuffer = 32 << 20,
  iterList = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 48, 64, 96, 128, 192, 256],
} = {}) {
  const cap = device.limits?.maxStorageBufferBindingSize ?? bytesPerBuffer;
  const size = Math.min(bytesPerBuffer, cap);
  const n = Math.floor(Math.floor(size / 16) / WG) * WG;
  const pipe = pipelineFor(device, SWEEP_WGSL);
  const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
  const a = device.createBuffer({ size: n * 16, usage: U });
  const o = device.createBuffer({ size: n * 16, usage: U | GPUBufferUsage.COPY_SRC });

  const points = [];
  for (const iters of iterList) {
    const uni = uniformBuf(device, [n, iters, 0, 0]);
    const bg = device.createBindGroup({
      layout: pipe.getBindGroupLayout(0),
      entries: [
        { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: a } },
        { binding: 2, resource: { buffer: o } },
      ],
    });
    const t = await timeDispatch(device, pipe, bg, n / WG, 5);
    uni.destroy();
    const flops = n * 4 * 4 * iters * 2;        // 4 lanes x 4 chains x 2 FLOP per FMA
    const bytes = n * 32;                       // 16 read + 16 written per vec4
    points.push({ iters, ai: flops / bytes, gflops: flops / t.medianNs, gbps: bytes / t.medianNs, ns: t.medianNs });
  }
  a.destroy(); o.destroy();

  const peak = Math.max(...points.map((p) => p.gflops));
  const knee = points.find((p) => p.gflops >= peak * 0.95);
  return { points, peakGflops: peak, ridgeAI: knee ? knee.ai : null };
}

/**
 * How many workgroups this GPU holds concurrently — the denominator of every
 * wave-quantization claim in the report's §3.2.
 *
 * WebGPU deliberately exposes no core count, no SM count and no occupancy, so this
 * cannot be looked up; on an A100 you would write 108 and move on. Measure it instead:
 * run identical fixed-cost workgroups at increasing counts and watch for the staircase.
 * While the count fits in one wave the dispatch takes constant time; the first step up
 * marks the capacity.
 *
 * Capacity is a property of the *shape* — occupancy falls as a kernel uses more
 * registers and shared memory — so the probe runs at workgroup_size(256) to match the
 * conv kernel's 16x16 geometry. Treat the answer as "capacity for a conv-shaped
 * workgroup", not as a universal constant for the device.
 */
export async function waveProbe(device, { iters = 4000, maxWG = 512 } = {}) {
  const pipe = pipelineFor(device, FMA_WGSL);
  const out = device.createBuffer({
    size: maxWG * WG * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC,
  });
  const uni = uniformBuf(device, [iters, maxWG * WG, 0, 0]);
  const kc = device.createBuffer({ size: 16, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  device.queue.writeBuffer(kc, 0, new Float32Array([1.0000001, 0.0000001, 0, 0]));
  const bg = device.createBindGroup({
    layout: pipe.getBindGroupLayout(0),
    entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: out } },
      { binding: 2, resource: { buffer: kc } },
    ],
  });

  // Ramp the clock before sampling. Without this the first few points come back at the
  // idle clock — the initial version read 1, 2 and 4 workgroups as *five times* slower
  // than 8, which inverted the whole curve and produced a capacity of 1.
  for (let i = 0; i < 4; i++) await timeDispatch(device, pipe, bg, maxWG, 1);

  const counts = [];
  for (let n = 1; n <= Math.min(96, maxWG); n++) counts.push(n);
  for (let n = 104; n <= maxWG; n += 8) counts.push(n);

  const points = [];
  for (const n of counts) {
    const t = await timeDispatch(device, pipe, bg, n, 5);
    points.push({ workgroups: n, ns: t.medianNs });
  }
  out.destroy(); uni.destroy(); kc.destroy();

  // Recover the staircase period by fitting, not by edge detection. If P workgroups run
  // concurrently, every n sharing a wave count ceil(n/P) should take the same time — so
  // group the points by ceil(n/P) and score how step-like that grouping is.
  //
  // Two rejected estimators, both of which got this wrong:
  //   - Edge detection. The plateaus carry ~15% ripples that no fixed threshold can
  //     separate from real steps; it reported half the true period.
  //   - R² of ns ~ a + b*ceil(n/P). Useless for discrimination: ceil(n/P) is nearly
  //     proportional to n for small P, and duration is nearly linear in n over the whole
  //     sweep, so P = 2, 3, 4, 5 and 10 all scored 1.000.
  //
  // An F-statistic is the right tool — it rewards tight plateaus but charges for the
  // extra groups a too-small period needs, which is exactly what separates P = 10 from
  // its own divisors.
  // Only the step-1 region can resolve the period at all: the sparse tail is sampled
  // every 8 workgroups, so it cannot distinguish a period of 10 from one of 5, and
  // because it holds a third of the points it otherwise dominates the statistic. It
  // stays in `points` for the report; it just doesn't get a vote here.
  const dense = points.filter((p) => p.workgroups <= 96);
  const score = (P) => {
    const groups = new Map();
    for (const pt of dense) {
      const k = Math.ceil(pt.workgroups / P);
      (groups.get(k) || groups.set(k, []).get(k)).push(pt.ns);
    }
    const N = dense.length, k = groups.size;
    if (k < 2 || k >= N) return 0;
    const grand = dense.reduce((s, p) => s + p.ns, 0) / N;
    let between = 0, within = 0;
    for (const vs of groups.values()) {
      const m = vs.reduce((s, v) => s + v, 0) / vs.length;
      between += vs.length * (m - grand) ** 2;
      for (const v of vs) within += (v - m) ** 2;
    }
    if (within <= 0) return Infinity;
    return (between / (k - 1)) / (within / (N - k));
  };
  let capacity = null, best = -1;
  const fits = [];
  for (let P = 2; P <= 128; P++) {
    const f = score(P);
    fits.push({ period: P, f });
    if (f > best) { best = f; capacity = P; }
  }
  const ranked = [...fits].sort((a, b) => b.f - a.f);
  // Runner-up margin, so a marginal win is visible rather than presented as certainty.
  const margin = ranked[1]?.f ? ranked[0].f / ranked[1].f : Infinity;
  return {
    points, capacity, f: best, margin, fits: ranked.slice(0, 5),
    note: margin > 1.3
      ? `staircase period fitted (F=${best.toFixed(0)}, ${margin.toFixed(1)}x clear of runner-up)`
      : `period ${capacity} only marginally beats ${ranked[1]?.period} — treat wave figures as approximate`,
  };
}

/** All four probes, plus the derived ridge point, as one JSON-serialisable block. */
export async function measureRoofs(device, opts = {}) {
  const compute = await computeRoof(device, opts.compute);
  const bandwidth = await bandwidthRoof(device, opts.bandwidth);
  const sweep = await intensitySweep(device, opts.sweep);
  const wave = await waveProbe(device, opts.wave);
  // Sanity check, not decoration. The FMA probe and the sweep's plateau measure the
  // same ceiling by different routes, so they should agree within a modest factor. When
  // they did not, the cause was the compiler folding the FMA loop — see the note above
  // FMA_WGSL. Anything that trips this invalidates every attainment number downstream,
  // so it travels with the results rather than being checked once and forgotten.
  const agreement = compute.gflops / sweep.peakGflops;
  return {
    compute, bandwidth, sweep, wave,
    waveCapacity: wave.capacity,
    // Two ridge estimates, deliberately both reported. The ratio is the textbook
    // definition; the swept one is what this machine actually does. They can differ a
    // lot on unified memory, and the swept value is the one to classify against.
    ridgeFromRatio: compute.gflops / bandwidth.gbps,
    ridgeFromSweep: sweep.ridgeAI,
    check: {
      computeVsSweepPlateau: agreement,
      ok: agreement > 0.7 && agreement < 1.5,
      note: "FMA roof and swept plateau should agree within ~1.5x; a large ratio means "
          + "the compiler folded the FMA loop and the compute roof is fiction.",
    },
  };
}
