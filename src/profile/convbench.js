// Convolution microbenchmark — the Phase 1 iteration loop.
//
// Phase 0 established that `conv` is 99.8% of GPU time and runs at ~3.5% of this
// machine's attainable roof, and that the cause is arithmetic throughput rather than
// memory traffic, occupancy or topology. So Phase 1 optimises exactly one kernel, and
// needs a loop faster than the ~10 s a full segmentation takes.
//
// This runs a single conv layer in isolation against a set of kernel variants, checks
// each variant's output against the shipping kernel numerically, and reports GFLOP/s
// against the measured roof. Every variant is scored on the *real* layer shapes weighted
// by their real share of conv time (SHAPES below), so a variant's aggregate score
// predicts what it will do to the whole pipeline instead of flattering it on one
// convenient size.
//
// Correctness is checked on every run and is not optional: a fast wrong kernel is worth
// nothing, and reordering FMAs is exactly the kind of change that silently breaks a
// weights-validated engine.

// Variants (including the frozen pre-Phase-1 reference) live in conv-variants.js; this
// file only knows how to run and score them.

// The cyto3 conv layer shapes, with each shape's measured share of total conv time from
// results/phase0-*.json (cellpose_020, 30 tiles). The top four are 74% of conv time and
// every shape but the 0.1% output head has Cout % 8 == 0 — which is what makes
// specialising on that case worth doing.
export const SHAPES = [
  { Cin: 256, Cout: 256, H: 28,  W: 28,  K: 3, share: 0.241 },
  { Cin: 128, Cout: 128, H: 56,  W: 56,  K: 3, share: 0.201 },
  { Cin: 64,  Cout: 64,  H: 112, W: 112, K: 3, share: 0.152 },
  { Cin: 32,  Cout: 32,  H: 224, W: 224, K: 3, share: 0.150 },
  { Cin: 256, Cout: 128, H: 56,  W: 56,  K: 3, share: 0.067 },
  { Cin: 128, Cout: 64,  H: 112, W: 112, K: 3, share: 0.051 },
  { Cin: 64,  Cout: 32,  H: 224, W: 224, K: 3, share: 0.050 },
  { Cin: 128, Cout: 256, H: 28,  W: 28,  K: 3, share: 0.017 },
  { Cin: 64,  Cout: 128, H: 56,  W: 56,  K: 3, share: 0.017 },
  { Cin: 32,  Cout: 64,  H: 112, W: 112, K: 3, share: 0.013 },
  { Cin: 256, Cout: 128, H: 56,  W: 56,  K: 1, share: 0.010 },
  { Cin: 128, Cout: 64,  H: 112, W: 112, K: 1, share: 0.008 },
  { Cin: 64,  Cout: 32,  H: 224, W: 224, K: 1, share: 0.008 },
  { Cin: 256, Cout: 256, H: 28,  W: 28,  K: 1, share: 0.006 },
  { Cin: 2,   Cout: 32,  H: 224, W: 224, K: 3, share: 0.002 },
  { Cin: 32,  Cout: 3,   H: 224, W: 224, K: 1, share: 0.001 },
];

const RESOLVE_USAGE = GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC;
const READ_USAGE = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;
const STORAGE = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;

// Deterministic pseudorandom fill: same data every run, so a variant's numbers are
// comparable across sessions and a correctness failure is reproducible.
function fill(n, seed) {
  const a = new Float32Array(n);
  let s = seed >>> 0;
  for (let i = 0; i < n; i++) {
    s = (s * 1664525 + 1013904223) >>> 0;
    a[i] = (s / 4294967296) * 2 - 1;
  }
  return a;
}

function makeShapeResources(device, shape) {
  const { Cin, Cout, H, W, K } = shape;
  const mk = (data) => {
    const b = device.createBuffer({ size: Math.max(256, data.length * 4), usage: STORAGE });
    device.queue.writeBuffer(b, 0, data);
    return b;
  };
  const inp = mk(fill(Cin * H * W, 1));
  const w = mk(fill(Cout * Cin * K * K, 2));
  const b = mk(fill(Cout, 3));
  // Keep scale/shift near identity so the output stays in a sane range and a relative
  // error comparison means something.
  const scale = mk(fill(Cin, 4).map((v) => 1 + v * 0.1));
  const shift = mk(fill(Cin, 5).map((v) => v * 0.1));
  const outp = device.createBuffer({ size: Math.max(256, Cout * H * W * 4), usage: STORAGE });
  const dummy = device.createBuffer({ size: 256, usage: STORAGE });
  const uni = device.createBuffer({
    size: 48, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
  });
  // Matches struct P in CONV_WGSL: relu on, no add, no resid — the common case.
  device.queue.writeBuffer(uni, 0, new Uint32Array(
    [H, W, Cin, Cout, K, (K / 2) | 0, 1, 0, 0, 0, 0, 0]));
  return { inp, w, b, scale, shift, outp, dummy, uni,
           destroy: () => [inp, w, b, scale, shift, outp, dummy, uni].forEach((x) => x.destroy()) };
}

function bindGroupFor(device, pipeline, r) {
  return device.createBindGroup({
    layout: pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: r.uni } },
      { binding: 1, resource: { buffer: r.inp } },
      { binding: 2, resource: { buffer: r.w } },
      { binding: 3, resource: { buffer: r.b } },
      { binding: 4, resource: { buffer: r.scale } },
      { binding: 5, resource: { buffer: r.shift } },
      { binding: 6, resource: { buffer: r.dummy } },
      { binding: 8, resource: { buffer: r.dummy } },
      { binding: 7, resource: { buffer: r.outp } },
    ],
  });
}

async function readback(device, buf, bytes) {
  const rb = device.createBuffer({ size: bytes, usage: READ_USAGE });
  const enc = device.createCommandEncoder();
  enc.copyBufferToBuffer(buf, 0, rb, 0, bytes);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const out = new Float32Array(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy();
  return out;
}

/** Median GPU duration of `reps` back-to-back dispatches, in nanoseconds. */
async function timeIt(device, pipeline, bg, wg, reps = 9) {
  const qs = device.createQuerySet({ type: "timestamp", count: reps * 2 });
  const res = device.createBuffer({ size: reps * 2 * 8, usage: RESOLVE_USAGE });
  const rb = device.createBuffer({ size: reps * 2 * 8, usage: READ_USAGE });
  const enc = device.createCommandEncoder({ label: "convbench" });
  for (let i = 0; i < reps; i++) {
    const p = enc.beginComputePass({
      timestampWrites: { querySet: qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1 },
    });
    p.setPipeline(pipeline); p.setBindGroup(0, bg);
    p.dispatchWorkgroups(wg[0], wg[1], wg[2]); p.end();
  }
  enc.resolveQuerySet(qs, 0, reps * 2, res, 0);
  enc.copyBufferToBuffer(res, 0, rb, 0, reps * 2 * 8);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(rb.getMappedRange().slice(0));
  rb.unmap(); rb.destroy(); res.destroy(); qs.destroy?.();
  const ns = [];
  // Skip the first: it absorbs the cold caches for this shape's buffers.
  for (let i = 1; i < reps; i++) ns.push(Number(t[i * 2 + 1] - t[i * 2]));
  ns.sort((a, b) => a - b);
  return ns[ns.length >> 1];
}

/** Largest relative difference between a variant's output and the reference's. */
function compare(ref, got) {
  let maxAbs = 0, maxRel = 0, scale = 0;
  for (let i = 0; i < ref.length; i++) scale = Math.max(scale, Math.abs(ref[i]));
  if (scale === 0) scale = 1;
  for (let i = 0; i < ref.length; i++) {
    const d = Math.abs(ref[i] - got[i]);
    if (d > maxAbs) maxAbs = d;
    const r = d / scale;
    if (r > maxRel) maxRel = r;
  }
  return { maxAbs, maxRel };
}

/**
 * Run every variant over every shape.
 *
 * @param roofGflops the measured compute roof, for attainment
 * @param tol        relative-error budget. Variants legitimately reorder FMAs, so exact
 *                   equality is the wrong bar; 1e-4 catches real bugs (wrong indexing,
 *                   dropped taps, off-by-one halos) while tolerating summation order.
 */
export async function benchVariants(device, variants, {
  shapes = SHAPES, roofGflops = null, reps = 9, tol = 1e-4,
} = {}) {
  // A variant's `wgsl` may be a function of the shape rather than a fixed string, so a
  // kernel can specialise on compile-time constants (K=3, Cout%8==0) the way an
  // integrated version would — the engine compiles one pipeline per specialisation and
  // picks at dispatch time. Pipelines are cached by generated source so a variant that
  // specialises on K builds two pipelines, not sixteen.
  const pipelines = new Map();
  const unavailable = new Map();   // variant id -> why it could not be built
  const pipelineFor = async (v, shape) => {
    const code = typeof v.wgsl === "function" ? v.wgsl(shape) : v.wgsl;
    if (!pipelines.has(code)) {
      // A variant can be legitimately unbuildable — the obvious one is exceeding
      // maxComputeWorkgroupStorageSize, which is 16 KB by default and is exactly what a
      // large input-channel staging factor spends. That is a result, not a crash.
      //
      // It has to be caught with an error scope, not try/catch: pipeline validation is
      // asynchronous, so createComputePipeline returns an *invalid* pipeline rather than
      // throwing, and the first synchronous symptom is getBindGroupLayout() failing
      // somewhere unrelated.
      device.pushErrorScope("validation");
      const pipe = device.createComputePipeline({
        layout: "auto",
        compute: { module: device.createShaderModule({ code, label: v.id }), entryPoint: "main" },
      });
      const err = await device.popErrorScope();
      pipelines.set(code, err ? null : pipe);
      if (err) unavailable.set(v.id, String(err.message).split("\n")[0].slice(0, 160));
    }
    return pipelines.get(code);
  };

  const rows = [];
  for (const shape of shapes) {
    const r = makeShapeResources(device, shape);
    const bytes = shape.Cout * shape.H * shape.W * 4;
    let ref = null;
    for (const v of variants) {
      if (v.applicable && !v.applicable(shape)) continue;
      const pipe = await pipelineFor(v, shape);
      if (!pipe) continue;
      const bg = bindGroupFor(device, pipe, r);
      const wg = v.dispatch(shape);

      // Correctness first, then timing — a variant that fails the check is not worth
      // timing and its number would only be misleading in the table.
      const enc = device.createCommandEncoder();
      const p = enc.beginComputePass(); p.setPipeline(pipe); p.setBindGroup(0, bg);
      p.dispatchWorkgroups(wg[0], wg[1], wg[2]); p.end();
      device.queue.submit([enc.finish()]);
      const out = await readback(device, r.outp, bytes);
      if (v.id === variants[0].id) ref = out;
      const cmp = ref === out ? { maxAbs: 0, maxRel: 0 } : compare(ref, out);
      const ok = cmp.maxRel <= tol;

      const ns = await timeIt(device, pipe, bg, wg, reps);
      const flops = 2 * shape.H * shape.W * shape.Cout * shape.Cin * shape.K * shape.K;
      const gflops = flops / ns;
      rows.push({
        variant: v.id, shape, ns, gflops, ok, ...cmp,
        workgroups: wg[0] * wg[1] * wg[2],
        attainment: roofGflops ? gflops / roofGflops : null,
        speedup: null,
      });
    }
    r.destroy();
  }

  // Speedup vs the first variant on the same shape, and the time-weighted aggregate that
  // predicts the effect on the full pipeline.
  const base = new Map();
  for (const row of rows) {
    if (row.variant === variants[0].id) base.set(JSON.stringify(row.shape), row.ns);
  }
  for (const row of rows) row.speedup = base.get(JSON.stringify(row.shape)) / row.ns;

  const summary = variants.map((v) => {
    const mine = rows.filter((r) => r.variant === v.id);
    if (!mine.length) {
      return {
        variant: v.id, what: v.what, unavailable: unavailable.get(v.id) || "no applicable shapes",
        covers: 0, allOk: false, failures: [],
        weightedGflops: null, weightedAttainment: null, projectedSpeedup: null,
      };
    }
    // Weights renormalise over the shapes this variant actually handles, and `covers`
    // reports how much of real conv time that is — a kernel that only does K=3 is not
    // comparable to one that does everything until you know it covers 95%.
    const wsum = mine.reduce((s, r) => s + r.shape.share, 0) || 1;
    // Weighted by time share, and harmonic in speedup — total time is the sum of per
    // shape times, so speedups do not average arithmetically.
    const projectedTime = mine.reduce((s, r) => s + r.shape.share / r.speedup, 0) / wsum;
    return {
      variant: v.id, what: v.what,
      covers: wsum / (shapes.reduce((s, x) => s + x.share, 0) || 1),
      allOk: mine.every((r) => r.ok),
      failures: mine.filter((r) => !r.ok).map((r) => `${r.shape.Cin}->${r.shape.Cout}@${r.shape.H}k${r.shape.K}`),
      weightedGflops: mine.reduce((s, r) => s + r.gflops * r.shape.share, 0) / wsum,
      weightedAttainment: roofGflops
        ? mine.reduce((s, r) => s + (r.gflops / roofGflops) * r.shape.share, 0) / wsum : null,
      projectedSpeedup: 1 / projectedTime,
    };
  });

  return { rows, summary };
}
