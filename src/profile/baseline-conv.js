// The pre-Phase-1 convolution kernel, frozen.
//
// This is the kernel src/cellpose.js shipped before Phase 1, kept verbatim as the
// reference that every variant is measured and diffed against. It is not dead code and
// must not be "cleaned up" into an import of the current kernel — the moment it tracks
// the shipping kernel, the Phase 1 waterfall loses its zero point and the numerical
// equivalence check compares the new kernel against itself.
//
// Measured here at ~104 GFLOP/s, 3.3% of this machine's roof.
export const BASELINE_CONV_WGSL = /* wgsl */`
const BLK = 8u;
const TS  = 16u;
const TW  = 18u;
// cp007: 'addv' is summed into the input before BN (skip connection); 'resid'
// is summed into the conv output (residual), fusing the elementwise residual-add
// passes into the conv that produces them (16 fewer dispatches per forward).
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

var<workgroup> tile: array<f32, TW * TW>;   // activated input tile for current ci
var<workgroup> ws:   array<f32, BLK * 9u>;  // weight slab for current ci

@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK;
  let nco = min(BLK, p.Cout - coBase);
  let HW = p.H * p.W;
  let K = p.K; let pad = i32(p.pad); let KK = K * K;
  let x = wg.x * TS + lid.x;
  let y = wg.y * TS + lid.y;
  let lt = lid.y * TS + lid.x;              // 0..255 flat thread id
  let ox = i32(wg.x * TS) - pad;            // tile origin (global) x
  let oy = i32(wg.y * TS) - pad;

  var acc: array<f32, BLK>;
  for (var j = 0u; j < BLK; j = j + 1u) { acc[j] = b[coBase + min(j, nco - 1u)]; }

  let tileN = TW * TW;
  let wN = nco * KK;
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
        for (var j = 0u; j < nco; j = j + 1u) {
          acc[j] = acc[j] + v * ws[j * KK + k];
        }
      }
    }
    workgroupBarrier();
  }
  if (x < p.W && y < p.H) {
    for (var j = 0u; j < nco; j = j + 1u) {
      let oi = (coBase + j) * HW + y * p.W + x;
      var o = acc[j];
      if (p.useResid == 1u) { o = o + resid[oi]; }
      outp[oi] = o;
    }
  }
}`;
