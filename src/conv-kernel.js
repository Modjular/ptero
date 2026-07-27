// The convolution kernel, generated.
//
// Phase 0 measured this kernel at ~3.5% of the machine's attainable roof while being
// 99.8% of all GPU time (docs/PHASE0.md). Phase 1 rebuilt it around three findings, each
// isolated and measured separately in tools/convbench.mjs:
//
//   1. Scalar accumulators — 5.0x.  The previous version accumulated into
//      `var acc: array<f32, BLK>` indexed by a loop whose bound came from a uniform. A
//      dynamically-indexed private array does not stay in registers; on Metal it spills
//      to thread-local memory, which is device-backed. That single property cost 5x.
//   2. Compile-time K — 1.4x on top. K came from the uniform, so the nine taps could not
//      be unrolled and every weight address was computed at runtime. K is 3 for ~95% of
//      conv time and 1 for the rest, so this ships as one pipeline per K.
//   3. Spatial register blocking — 1.6x on top. One shared-memory read per FMA caps
//      throughput at shared-memory bandwidth. Each thread now computes an RBY x RBX block
//      of output pixels, so a weight read feeds RBY*RBX FMAs.
//
// Together: 10.9x, 3.5% -> 37% of roof, and **bit-identical output** to the kernel it
// replaces (maxRel 0.0 across every layer shape in the benchmark) — the FMA order per
// output element is unchanged, only which thread performs it.
//
// The block size is a measured optimum, not a maximum: 2x4 loses to 2x2 because 64 live
// accumulators cost more occupancy than they save in shared-memory traffic. Re-measure
// with `node tools/convbench.mjs` before changing it.
//
// Cout is a multiple of BLK for every layer but the 3-channel output head, so writes stay
// guarded rather than assuming it.

// All four are measured optima from a joint sweep, not independent choices — they trade
// against each other through two shared budgets, registers (BLK*RBY*RBX accumulators) and
// threadgroup memory (CB tiles). Raising any one past these values *loses*: BLK=32 at
// 2x2 collapses to 16% of roof on register pressure, and CB=8 exceeds even the 32 KB
// budget. Re-run `node tools/convbench.mjs` before touching them.
export const BLK = 16;   // output channels per workgroup — divides activation re-reads
export const TS = 16;    // workgroup side, in threads
export const RBY = 2;    // output pixels per thread, vertical   } divides shared-memory
export const RBX = 2;    // output pixels per thread, horizontal } weight reads per FMA
export const CB = 4;     // input channels staged per barrier round — divides barriers

/** Workgroup grid for one conv dispatch. Must match the shader's geometry. */
export const convDispatch = (H, W, Cout, rby = RBY, rbx = RBX, blk = BLK) =>
  [Math.ceil(W / (TS * rbx)), Math.ceil(H / (TS * rby)), Math.ceil(Cout / blk)];

/**
 * Generate the conv shader for a given kernel size.
 *
 * `useAdd` (skip connection summed into the input before BN) and `useResid` (residual
 * summed into the output) stay uniform-driven rather than specialised: they do not affect
 * the inner loop, and specialising them would quadruple the pipeline count for nothing.
 */
export function convWGSL(K, rby = RBY, rbx = RBX, blk = BLK, cb = CB) {
  const KK = K * K, PAD = (K / 2) | 0;
  const OUTX = TS * rbx, OUTY = TS * rby;
  const TWx = OUTX + (K - 1), TWy = OUTY + (K - 1);
  const TWxy = TWx * TWy;
  const P = rby * rbx;
  const acc = (c, p) => `a${c}_${p}`;
  const chans = Array.from({ length: blk }, (_, c) => c);
  const pixels = Array.from({ length: P }, (_, p) => p);

  // The tap/FMA body, emitted once per staged input channel. Reading each activation
  // once and reusing it across all BLK output channels is the register-blocking win;
  // unrolling over the staged channels on top of that gives the scheduler `cb`
  // independent FMA chains to interleave while a shared-memory read is in flight.
  const inner = (s) => {
    const tOff = s === 0 ? "" : ` + ${s * TWxy}u`;
    const wOff = s === 0 ? "" : ` + ${s * blk * KK}u`;
    const taps = [];
    for (let py = 0; py < rby; py++) {
      for (let px = 0; px < rbx; px++) {
        taps.push(`        let v${s}_${py * rbx + px} = tile[(ly + ${py}u + ky) * TWx + lx + ${px}u + kx${tOff}];`);
      }
    }
    const fmas = chans.flatMap((c) => [
      `        let w${s}_${c} = ws[${c}u * KK + k${wOff}];`,
      ...pixels.map((p) => `        ${acc(c, p)} = ${acc(c, p)} + v${s}_${p} * w${s}_${c};`),
    ]);
    return [...taps, ...fmas].join("\n");
  };
  const stages = Array.from({ length: cb }, (_, s) => s);

  const stores = [];
  for (let py = 0; py < rby; py++) {
    for (let px = 0; px < rbx; px++) {
      const p = py * rbx + px;
      stores.push(`  {
    let oy2 = by + ${py}u; let ox2 = bx + ${px}u;
    if (ox2 < p.W && oy2 < p.H) {
      let o = oy2 * p.W + ox2;
${chans.map((c) => `      if (coBase + ${c}u < p.Cout) { let oi = (coBase + ${c}u) * HW + o; var t = ${acc(c, p)}; if (p.useResid == 1u) { t = t + resid[oi]; } outp[oi] = t; }`).join("\n")}
    }
  }`);
    }
  }

  return /* wgsl */`
// generated by src/conv-kernel.js — K=${K}, register block ${rby}x${rbx}
struct P { H:u32, W:u32, Cin:u32, Cout:u32, K:u32, pad:u32, useRelu:u32, useAdd:u32,
           useResid:u32, _p0:u32, _p1:u32, _p2:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       inp:   array<f32>;
@group(0) @binding(2) var<storage,read>       w:     array<f32>;
@group(0) @binding(3) var<storage,read>       b:     array<f32>;
@group(0) @binding(4) var<storage,read>       scale: array<f32>;
@group(0) @binding(5) var<storage,read>       shift: array<f32>;
@group(0) @binding(6) var<storage,read>       addv:  array<f32>;
@group(0) @binding(8) var<storage,read>       resid: array<f32>;
@group(0) @binding(7) var<storage,read_write> outp:  array<f32>;
const TS  = ${TS}u;
const BLK = ${blk}u;
const K   = ${K}u;
const KK  = ${KK}u;
const PAD = ${PAD};
const TWx = ${TWx}u;
const TWy = ${TWy}u;
var<workgroup> tile: array<f32, ${cb * TWxy}u>;
var<workgroup> ws:   array<f32, ${cb * blk * KK}u>;

@compute @workgroup_size(${TS},${TS},1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK;
  let HW = p.H * p.W;
  let lt = lid.y * TS + lid.x;
  let lx = lid.x * ${rbx}u;
  let ly = lid.y * ${rby}u;
  let bx = wg.x * ${OUTX}u + lx;
  let by = wg.y * ${OUTY}u + ly;
  let ox = i32(wg.x * ${OUTX}u) - PAD;
  let oy = i32(wg.y * ${OUTY}u) - PAD;

${chans.map((c) => `  let bc${c} = b[coBase + ${c}u];`).join("\n")}
${chans.flatMap((c) => pixels.map((p) => `  var ${acc(c, p)} = bc${c};`)).join("\n")}

  let stride = p.Cin * KK;
  // Input channels are staged CB at a time per barrier round. The barrier count is the
  // reason: at Cin=256 with CB=1 a workgroup pays 512 of them, and each is a pipeline
  // drain that the ~3 resident waves cannot hide.
  for (var ci = 0u; ci < p.Cin; ci = ci + ${cb}u) {
    // Cooperative load of the staged activation tiles, applying the optional skip add,
    // the BN scale/shift and the optional relu on the way in — the fusion that makes the
    // elementwise operators free here (Phase 0 measured all non-conv work at 0.19%).
    for (var i = lt; i < ${cb * TWxy}u; i = i + TS * TS) {
      let s = i / ${TWxy}u;
      let j = i % ${TWxy}u;
      let ty = j / TWx; let tx = j % TWx;
      let gy = oy + i32(ty); let gx = ox + i32(tx);
      var v = 0.0;
      // Cin is not always a multiple of cb (the stem has Cin=2), so a staged channel
      // past the end contributes a zero activation rather than being special-cased.
      if (ci + s < p.Cin && gy >= 0 && gy < i32(p.H) && gx >= 0 && gx < i32(p.W)) {
        let idx = (ci + s) * HW + u32(gy) * p.W + u32(gx);
        v = inp[idx];
        if (p.useAdd == 1u) { v = v + addv[idx]; }
        v = v * scale[ci + s] + shift[ci + s];
        if (p.useRelu == 1u) { v = max(v, 0.0); }
      }
      tile[i] = v;
    }
    for (var i = lt; i < ${cb * blk * KK}u; i = i + TS * TS) {
      let s = i / ${blk * KK}u;
      let j = (i % ${blk * KK}u) / KK;
      let kk = i % KK;
      var wv = 0.0;
      if (ci + s < p.Cin) { wv = w[(coBase + j) * stride + (ci + s) * KK + kk]; }
      ws[i] = wv;
    }
    workgroupBarrier();
    for (var ky = 0u; ky < K; ky = ky + 1u) {
      for (var kx = 0u; kx < K; kx = kx + 1u) {
        let k = ky * K + kx;
${stages.map(inner).join("\n")}
      }
    }
    workgroupBarrier();
  }
${stores.join("\n")}
}`;
}
