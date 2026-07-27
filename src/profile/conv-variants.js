// Conv kernel variants under test in Phase 1.
//
// Phase 0's finding: `conv` is 99.8% of GPU time at ~3.5% of the attainable roof, and
// the loss is arithmetic throughput, not bandwidth (99.8% `kernel-throughput`, 0.0%
// `traffic-amplification`). docs/PHASE0.md lists three hypotheses; these variants test
// them one at a time so the waterfall attributes the win to a cause rather than to a
// rewrite.
//
// Every variant keeps the shipping kernel's binding layout and uniform struct, so
// convbench.js can swap them under one bind group and compare outputs numerically.
//
// Shape facts that make specialisation legitimate (measured, from the Phase 0 results):
//   - K=3 is ~95% of conv time, K=1 the rest. Both must work; only K=3 needs to be fast.
//   - Every shape except the 0.1% output head has Cout % 8 == 0.
// Writes stay guarded by `coBase + j < p.Cout` regardless, so a specialised kernel is
// still *correct* on the ragged shape, just does wasted work there.

import { BASELINE_CONV_WGSL } from "./baseline-conv.js";
import { convWGSL, convDispatch } from "../conv-kernel.js";

const TS = 16;   // spatial tile side, = workgroup x/y
const BLK = 8;   // output channels per workgroup

// Shared declarations. Identical bindings to the shipping kernel — binding 8 sits after
// 7 because that is how CONV_WGSL grew, and changing it would break the bind group.
const HEADER = (tw, extra = "") => /* wgsl */`
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
const BLK = ${BLK}u;
const TW  = ${tw}u;
var<workgroup> tile: array<f32, TW * TW>;
${extra}`;

// ---------------------------------------------------------------------------
// v1 — scalar accumulators.
//
// Tests docs/PHASE0.md hypothesis 1. The shipping kernel accumulates into
// `var acc: array<f32, BLK>` indexed by a loop variable whose bound `nco` is computed
// from a uniform. A dynamically-indexed private array is the classic case where a
// backend gives up on registers and spills to thread-local memory, which on Metal is
// device-backed — that would cost two round trips per FMA and would explain the whole
// result on its own.
//
// Change, and only this change: eight named scalars and a fully unrolled inner loop, so
// nothing is dynamically indexed. Tile loading, barriers, loop structure and arithmetic
// order are otherwise identical to the baseline.
// ---------------------------------------------------------------------------
const V1 = /* wgsl */`
${HEADER(TS + 2, "var<workgroup> ws: array<f32, BLK * 9u>;")}
@compute @workgroup_size(${TS},${TS},1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK;
  let HW = p.H * p.W;
  let K = p.K; let pad = i32(p.pad); let KK = K * K;
  let x = wg.x * TS + lid.x;
  let y = wg.y * TS + lid.y;
  let lt = lid.y * TS + lid.x;
  let ox = i32(wg.x * TS) - pad;
  let oy = i32(wg.y * TS) - pad;

  var a0 = b[coBase + 0u]; var a1 = b[coBase + 1u];
  var a2 = b[coBase + 2u]; var a3 = b[coBase + 3u];
  var a4 = b[coBase + 4u]; var a5 = b[coBase + 5u];
  var a6 = b[coBase + 6u]; var a7 = b[coBase + 7u];

  let tileN = TW * TW;
  let wN = BLK * KK;
  let stride = p.Cin * KK;
  for (var ci = 0u; ci < p.Cin; ci = ci + 1u) {
    let base = ci * HW;
    let sc = scale[ci]; let sh = shift[ci];
    for (var i = lt; i < tileN; i = i + TS * TS) {
      let ty = i / TW; let tx = i % TW;
      let gy = oy + i32(ty); let gx = ox + i32(tx);
      var v = 0.0;
      if (gy >= 0 && gy < i32(p.H) && gx >= 0 && gx < i32(p.W)) {
        let idx = base + u32(gy) * p.W + u32(gx);
        v = inp[idx];
        if (p.useAdd == 1u) { v = v + addv[idx]; }
        v = v * sc + sh;
        if (p.useRelu == 1u) { v = max(v, 0.0); }
      }
      tile[i] = v;
    }
    let wbase = ci * KK;
    for (var i = lt; i < wN; i = i + TS * TS) {
      let j = i / KK; let k = i % KK;
      ws[i] = w[(coBase + j) * stride + wbase + k];
    }
    workgroupBarrier();
    for (var ky = 0u; ky < K; ky = ky + 1u) {
      for (var kx = 0u; kx < K; kx = kx + 1u) {
        let v = tile[(lid.y + ky) * TW + (lid.x + kx)];
        let k = ky * K + kx;
        a0 = a0 + v * ws[0u * KK + k]; a1 = a1 + v * ws[1u * KK + k];
        a2 = a2 + v * ws[2u * KK + k]; a3 = a3 + v * ws[3u * KK + k];
        a4 = a4 + v * ws[4u * KK + k]; a5 = a5 + v * ws[5u * KK + k];
        a6 = a6 + v * ws[6u * KK + k]; a7 = a7 + v * ws[7u * KK + k];
      }
    }
    workgroupBarrier();
  }
  if (x < p.W && y < p.H) {
    let o = y * p.W + x;
    if (coBase + 0u < p.Cout) { var t = a0; if (p.useResid == 1u) { t = t + resid[(coBase + 0u) * HW + o]; } outp[(coBase + 0u) * HW + o] = t; }
    if (coBase + 1u < p.Cout) { var t = a1; if (p.useResid == 1u) { t = t + resid[(coBase + 1u) * HW + o]; } outp[(coBase + 1u) * HW + o] = t; }
    if (coBase + 2u < p.Cout) { var t = a2; if (p.useResid == 1u) { t = t + resid[(coBase + 2u) * HW + o]; } outp[(coBase + 2u) * HW + o] = t; }
    if (coBase + 3u < p.Cout) { var t = a3; if (p.useResid == 1u) { t = t + resid[(coBase + 3u) * HW + o]; } outp[(coBase + 3u) * HW + o] = t; }
    if (coBase + 4u < p.Cout) { var t = a4; if (p.useResid == 1u) { t = t + resid[(coBase + 4u) * HW + o]; } outp[(coBase + 4u) * HW + o] = t; }
    if (coBase + 5u < p.Cout) { var t = a5; if (p.useResid == 1u) { t = t + resid[(coBase + 5u) * HW + o]; } outp[(coBase + 5u) * HW + o] = t; }
    if (coBase + 6u < p.Cout) { var t = a6; if (p.useResid == 1u) { t = t + resid[(coBase + 6u) * HW + o]; } outp[(coBase + 6u) * HW + o] = t; }
    if (coBase + 7u < p.Cout) { var t = a7; if (p.useResid == 1u) { t = t + resid[(coBase + 7u) * HW + o]; } outp[(coBase + 7u) * HW + o] = t; }
  }
}`;

// ---------------------------------------------------------------------------
// v2 — v1 plus K as a compile-time constant.
//
// In the shipping kernel K comes from the uniform, so the ky/kx tap loops have runtime
// trip counts, `KK` is a runtime value, and every `ws[j*KK+k]` is a runtime address
// computation. K is 3 for ~95% of conv time and 1 for the rest, so specialising costs
// two pipelines and lets the backend unroll all nine taps and fold the addressing.
//
// Generated per K, which is why `wgsl` here is a function of the shape.
// ---------------------------------------------------------------------------
const V2 = (K) => /* wgsl */`
${HEADER(TS + (K - 1), "var<workgroup> ws: array<f32, BLK * " + (K * K) + "u>;")}
const K  = ${K}u;
const KK = ${K * K}u;
const PAD = ${(K / 2) | 0};
@compute @workgroup_size(${TS},${TS},1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK;
  let HW = p.H * p.W;
  let x = wg.x * TS + lid.x;
  let y = wg.y * TS + lid.y;
  let lt = lid.y * TS + lid.x;
  let ox = i32(wg.x * TS) - PAD;
  let oy = i32(wg.y * TS) - PAD;

  var a0 = b[coBase + 0u]; var a1 = b[coBase + 1u];
  var a2 = b[coBase + 2u]; var a3 = b[coBase + 3u];
  var a4 = b[coBase + 4u]; var a5 = b[coBase + 5u];
  var a6 = b[coBase + 6u]; var a7 = b[coBase + 7u];

  let stride = p.Cin * KK;
  for (var ci = 0u; ci < p.Cin; ci = ci + 1u) {
    let base = ci * HW;
    let sc = scale[ci]; let sh = shift[ci];
    for (var i = lt; i < TW * TW; i = i + TS * TS) {
      let ty = i / TW; let tx = i % TW;
      let gy = oy + i32(ty); let gx = ox + i32(tx);
      var v = 0.0;
      if (gy >= 0 && gy < i32(p.H) && gx >= 0 && gx < i32(p.W)) {
        let idx = base + u32(gy) * p.W + u32(gx);
        v = inp[idx];
        if (p.useAdd == 1u) { v = v + addv[idx]; }
        v = v * sc + sh;
        if (p.useRelu == 1u) { v = max(v, 0.0); }
      }
      tile[i] = v;
    }
    let wbase = ci * KK;
    for (var i = lt; i < BLK * KK; i = i + TS * TS) {
      let j = i / KK; let k = i % KK;
      ws[i] = w[(coBase + j) * stride + wbase + k];
    }
    workgroupBarrier();
    for (var ky = 0u; ky < K; ky = ky + 1u) {
      for (var kx = 0u; kx < K; kx = kx + 1u) {
        let v = tile[(lid.y + ky) * TW + (lid.x + kx)];
        let k = ky * K + kx;
        a0 = a0 + v * ws[0u * KK + k]; a1 = a1 + v * ws[1u * KK + k];
        a2 = a2 + v * ws[2u * KK + k]; a3 = a3 + v * ws[3u * KK + k];
        a4 = a4 + v * ws[4u * KK + k]; a5 = a5 + v * ws[5u * KK + k];
        a6 = a6 + v * ws[6u * KK + k]; a7 = a7 + v * ws[7u * KK + k];
      }
    }
    workgroupBarrier();
  }
  if (x < p.W && y < p.H) {
    let o = y * p.W + x;
    for (var j = 0u; j < BLK; j = j + 1u) {
      if (coBase + j >= p.Cout) { break; }
      var t = select(select(select(select(select(select(select(a7, a6, j == 6u), a5, j == 5u), a4, j == 4u), a3, j == 3u), a2, j == 2u), a1, j == 1u), a0, j == 0u);
      let oi = (coBase + j) * HW + o;
      if (p.useResid == 1u) { t = t + resid[oi]; }
      outp[oi] = t;
    }
  }
}`;

// ---------------------------------------------------------------------------
// v3 — spatial register blocking, on top of v2.
//
// Tests docs/PHASE0.md hypothesis 2. After v2 the kernel still issues one shared-memory
// read per FMA: per (ky,kx) each thread reads one activation from `tile` and eight
// weights from `ws` to do eight FMAs. That caps throughput at shared-memory bandwidth
// however good the ALU pipeline is.
//
// Fix: give each thread an RBY x RBX block of output pixels instead of one. A weight
// read from `ws` then feeds RBY*RBX FMAs rather than one, and because the blocks are
// contiguous the activation neighbourhoods overlap too. At 2x2 the ratio goes from
// 32 FMA : 40 shared reads to 32 FMA : 12.
//
// The cost is registers: BLK * RBY * RBX accumulators live across the whole Cin loop
// (32 of them at 2x2). Too many and occupancy falls, which is why the block size is a
// parameter to be swept rather than a constant to be maximised.
//
// Generated rather than written out, because 32 named accumulators by hand is how
// off-by-one bugs get in.
// ---------------------------------------------------------------------------
const V3 = (K, RBY, RBX) => {
  const KK = K * K, PAD = (K / 2) | 0;
  const OUT = TS * RBX, OUTY = TS * RBY;   // output tile covered by one workgroup
  const TWx = OUT + (K - 1), TWy = OUTY + (K - 1);
  const P = RBY * RBX;
  const accName = (c, p) => `a${c}_${p}`;

  const decls = [];
  for (let c = 0; c < BLK; c++) {
    for (let p = 0; p < P; p++) decls.push(`var ${accName(c, p)} = bc${c};`);
  }

  // Per tap: read the RBY*RBX activations once, then reuse each across all BLK channels.
  const taps = [];
  for (let py = 0; py < RBY; py++) {
    for (let px = 0; px < RBX; px++) {
      taps.push(`      let v${py * RBX + px} = tile[(ly + ${py}u + ky) * TWx + lx + ${px}u + kx];`);
    }
  }
  const fmas = [];
  for (let c = 0; c < BLK; c++) {
    fmas.push(`      let wv${c} = ws[${c}u * KK + k];`);
    for (let p = 0; p < P; p++) {
      fmas.push(`      ${accName(c, p)} = ${accName(c, p)} + v${p} * wv${c};`);
    }
  }

  const stores = [];
  for (let py = 0; py < RBY; py++) {
    for (let px = 0; px < RBX; px++) {
      const p = py * RBX + px;
      stores.push(`  {
    let oy2 = by + ${py}u; let ox2 = bx + ${px}u;
    if (ox2 < p.W && oy2 < p.H) {
      let o = oy2 * p.W + ox2;
${Array.from({ length: BLK }, (_, c) => `      if (coBase + ${c}u < p.Cout) { let oi = (coBase + ${c}u) * HW + o; var t = ${accName(c, p)}; if (p.useResid == 1u) { t = t + resid[oi]; } outp[oi] = t; }`).join("\n")}
    }
  }`);
    }
  }

  return /* wgsl */`
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
const BLK = ${BLK}u;
const K   = ${K}u;
const KK  = ${KK}u;
const PAD = ${PAD};
const TWx = ${TWx}u;
const TWy = ${TWy}u;
var<workgroup> tile: array<f32, ${TWx * TWy}u>;
var<workgroup> ws:   array<f32, ${BLK * KK}u>;

@compute @workgroup_size(${TS},${TS},1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK;
  let HW = p.H * p.W;
  let lt = lid.y * TS + lid.x;
  let lx = lid.x * ${RBX}u;
  let ly = lid.y * ${RBY}u;
  let bx = wg.x * ${OUT}u + lx;
  let by = wg.y * ${OUTY}u + ly;
  let ox = i32(wg.x * ${OUT}u) - PAD;
  let oy = i32(wg.y * ${OUTY}u) - PAD;

${Array.from({ length: BLK }, (_, c) => `  let bc${c} = b[coBase + ${c}u];`).join("\n")}
${decls.map((d) => "  " + d).join("\n")}

  let stride = p.Cin * KK;
  for (var ci = 0u; ci < p.Cin; ci = ci + 1u) {
    let base = ci * HW;
    let sc = scale[ci]; let sh = shift[ci];
    for (var i = lt; i < TWx * TWy; i = i + TS * TS) {
      let ty = i / TWx; let tx = i % TWx;
      let gy = oy + i32(ty); let gx = ox + i32(tx);
      var v = 0.0;
      if (gy >= 0 && gy < i32(p.H) && gx >= 0 && gx < i32(p.W)) {
        let idx = base + u32(gy) * p.W + u32(gx);
        v = inp[idx];
        if (p.useAdd == 1u) { v = v + addv[idx]; }
        v = v * sc + sh;
        if (p.useRelu == 1u) { v = max(v, 0.0); }
      }
      tile[i] = v;
    }
    let wbase = ci * KK;
    for (var i = lt; i < BLK * KK; i = i + TS * TS) {
      let j = i / KK; let kk = i % KK;
      ws[i] = w[(coBase + j) * stride + wbase + kk];
    }
    workgroupBarrier();
    for (var ky = 0u; ky < K; ky = ky + 1u) {
      for (var kx = 0u; kx < K; kx = kx + 1u) {
        let k = ky * K + kx;
${taps.join("\n")}
${fmas.join("\n")}
      }
    }
    workgroupBarrier();
  }
${stores.join("\n")}
}`;
};

// ---------------------------------------------------------------------------
// v4 — v3 with half-width shared memory.
//
// Tests the storage half of docs/PHASE0.md hypothesis 3. Register blocking cut the
// number of shared-memory reads; this cuts their width, holding `tile` and `ws` as f16
// and widening back to f32 for the arithmetic.
//
// Accumulation stays f32 deliberately. These weights are validated against a desktop
// reference and the engine's whole claim is that it reproduces it; f16 accumulation over
// a 256-channel reduction would put that at risk to chase a kernel that is not obviously
// ALU-bound anyway. Storage-only f16 keeps every FMA in f32, so the numerical change is
// limited to rounding the inputs.
// ---------------------------------------------------------------------------
const V4 = (K, RBY, RBX) => "enable f16;\n" + V3(K, RBY, RBX)
  .replace("var<workgroup> tile: array<f32,", "var<workgroup> tile: array<f16,")
  .replace("var<workgroup> ws:   array<f32,", "var<workgroup> ws:   array<f16,")
  .replace("      tile[i] = v;", "      tile[i] = f16(v);")
  .replace(/ws\[i\] = w\[\(coBase \+ j\) \* stride \+ wbase \+ kk\];/,
           "ws[i] = f16(w[(coBase + j) * stride + wbase + kk]);")
  // Widen at the point of use so every multiply-add below stays f32.
  .replace(/let (v\d+) = tile\[([^\]]+)\];/g, "let $1 = f32(tile[$2]);")
  .replace(/let (wv\d+) = ws\[([^\]]+)\];/g, "let $1 = f32(ws[$2]);");

const rbDispatch = (RBY, RBX) => ({ H, W, Cout }) =>
  [Math.ceil(W / (TS * RBX)), Math.ceil(H / (TS * RBY)), Math.ceil(Cout / BLK)];

export const VARIANTS = [
  {
    id: "baseline",
    what: "the pre-Phase-1 kernel — the reference and the zero point",
    wgsl: BASELINE_CONV_WGSL,
    dispatch: ({ H, W, Cout }) =>
      [Math.ceil(W / TS), Math.ceil(H / TS), Math.ceil(Cout / BLK)],
  },
  {
    // Keeps the benchmark honest about what actually ships: if this ever diverges from
    // regblk_2x2 below, the engine and the experiment have drifted apart.
    id: "shipping",
    what: "the kernel in src/cellpose.js today",
    wgsl: ({ K }) => convWGSL(K),
    dispatch: ({ H, W, Cout }) => convDispatch(H, W, Cout),
  },
  {
    id: "scalar_acc",
    what: "H1: 8 scalar accumulators, no dynamically-indexed private array",
    wgsl: V1,
    dispatch: ({ H, W, Cout }) =>
      [Math.ceil(W / TS), Math.ceil(H / TS), Math.ceil(Cout / BLK)],
  },
  {
    id: "scalar_acc_k",
    what: "H1 + K as a compile-time constant (one pipeline per K)",
    wgsl: ({ K }) => V2(K),
    dispatch: ({ H, W, Cout }) =>
      [Math.ceil(W / TS), Math.ceil(H / TS), Math.ceil(Cout / BLK)],
  },
  {
    id: "regblk_2x2_f16",
    what: "f16 shared memory, f32 accumulation — rejected on numerics, see docs/PHASE1.md",
    wgsl: ({ K }) => V4(K, 2, 2),
    dispatch: rbDispatch(2, 2),
  },

  // ---------------------------------------------------------------------------
  // The (BLK, RBY, RBX) sweep.
  //
  // These three interact and cannot be tuned separately, which is why they are swept as
  // a grid rather than one at a time:
  //
  //   BLK  output channels per workgroup. Divides how many times the activation tile is
  //        re-read from global memory — the grid's z extent is ceil(Cout/BLK), and every
  //        z-block walks all of Cin. Raising it is the direct fix for the amplification
  //        the cost model reports (4-14x on the K=3 shapes).
  //   RBY,RBX  output pixels per thread. Divides how many shared-memory weight reads it
  //        takes to do a given number of FMAs.
  //
  // Both are bought with the same currency: BLK*RBY*RBX live accumulators per thread.
  // The rows below hold that product at 16, 32 and 64 so the *split* is what varies,
  // which is the question — 2x4 already showed that simply raising the product loses.
  // ---------------------------------------------------------------------------
  ...[
    [8, 1, 2, 1], [16, 1, 1, 1], [4, 2, 2, 1],                          // 16 accumulators
    [8, 2, 2, 1], [16, 2, 1, 1], [16, 1, 2, 1], [32, 1, 1, 1],          // 32
    [16, 2, 2, 1], [32, 1, 2, 1], [32, 2, 1, 1], [64, 1, 1, 1], [8, 2, 4, 1], // 64
    [32, 2, 2, 1], [64, 1, 2, 1],                                        // 128
    // Input-channel staging on the best of the above. Trades shared memory for barrier
    // count: CB=4 at 2x2 needs ~19 KB of the 32 KB threadgroup budget, which may cost
    // more occupancy than the barriers were costing.
    [8, 2, 2, 2], [8, 2, 2, 4], [16, 2, 2, 2], [16, 2, 2, 4], [16, 2, 2, 8],
    [16, 1, 2, 2], [16, 1, 2, 4], [16, 1, 2, 8], [32, 1, 2, 2], [32, 1, 2, 4],
    [8, 1, 2, 4], [8, 1, 2, 8], [32, 2, 2, 2], [8, 2, 2, 8],
  ].map(([blk, rby, rbx, cb]) => ({
    id: `blk${blk}_rb${rby}x${rbx}${cb > 1 ? `_cb${cb}` : ""}`,
    what: `BLK=${blk}, ${rby}x${rbx} block, CB=${cb} — ${blk * rby * rbx} accumulators`,
    wgsl: ({ K }) => convWGSL(K, rby, rbx, blk, cb),
    dispatch: ({ H, W, Cout }) => convDispatch(H, W, Cout, rby, rbx, blk),
  })),
];
