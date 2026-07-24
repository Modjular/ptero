// StarDist (2D_versatile_fluo) inference on WebGPU — environment-agnostic core.
// Sibling to cellpose_core.js: runs in Deno (headless tests) and the browser.
// The U-Net forward is hand-written WGSL; the polygon NMS post-process runs in JS.
//
// Public API:
//   const sd = await StarDistWebGPU.create();
//   sd.loadWeights(manifestObj, binArrayBuffer);
//   const {prob, dist, gh, gw} = await sd.forwardFromInput(inputF32, Hp, Wp);
//   const labels = sd.computeMasks(prob, dist, gh, gw, Hp, Wp, opts);  // Int32Array
//   const {labels, timings} = await sd.segmentImage(gray, H, W, opts);
//
// StarDist vs Cellpose (see cellpose_core.js): no flow field and no iterative
// dynamics — the U-Net emits, per grid pixel, an object probability + n_rays=32
// radial distances to the boundary (a whole star-convex polygon), and the only
// post-process is greedy non-maximum suppression between those polygons. The net
// is also a plain U-Net (no BatchNorm, no style vector), so the conv shader below
// is the cellpose tiled conv stripped of BN/skip/residual, with ReLU applied to
// the conv *output* (StarDist is conv->ReLU, where cellpose folds BN pre-conv).

const N_RAYS = 32;
const BLK = 8;          // output channels accumulated per workgroup (register block)
const TS = 16;          // tile side
const TW = TS + 2;      // tile side incl. halo (max pad = 1 for K∈{1,3})

// Shared-memory tiled conv (per input channel), same footprint as cellpose cp004
// (~1.6 KB shared, high occupancy). A 16×16 workgroup loads, per input channel, an
// 18×18 input tile + this channel's BLK×K×K weight slab, and all 256 threads
// accumulate BLK output channels. `useRelu` clamps the output (post-conv ReLU).
const CONV_WGSL = /* wgsl */`
const BLK = ${BLK}u;
const TS  = ${TS}u;
const TW  = ${TW}u;
struct P { H:u32, W:u32, Cin:u32, Cout:u32, K:u32, pad:u32, useRelu:u32, _p:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       inp:  array<f32>;
@group(0) @binding(2) var<storage,read>       w:    array<f32>;
@group(0) @binding(3) var<storage,read>       b:    array<f32>;
@group(0) @binding(4) var<storage,read_write> outp: array<f32>;

var<workgroup> tile: array<f32, TW * TW>;
var<workgroup> ws:   array<f32, BLK * 9u>;

@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>,
        @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK;
  let nco = min(BLK, p.Cout - coBase);
  let HW = p.H * p.W;
  let K = p.K; let pad = i32(p.pad); let KK = K * K;
  let x = wg.x * TS + lid.x;
  let y = wg.y * TS + lid.y;
  let lt = lid.y * TS + lid.x;
  let ox = i32(wg.x * TS) - pad;
  let oy = i32(wg.y * TS) - pad;

  var acc: array<f32, BLK>;
  for (var j = 0u; j < BLK; j = j + 1u) { acc[j] = b[coBase + min(j, nco - 1u)]; }

  let tileN = TW * TW;
  let wN = nco * KK;
  let stride = p.Cin * KK;
  for (var ci = 0u; ci < p.Cin; ci = ci + 1u) {
    let base = ci * HW;
    for (var i = lt; i < tileN; i = i + TS * TS) {
      let ty = i / TW; let tx = i % TW;
      let gy = oy + i32(ty); let gx = ox + i32(tx);
      var v = 0.0;
      if (gy >= 0 && gy < i32(p.H) && gx >= 0 && gx < i32(p.W)) {
        v = inp[base + u32(gy) * p.W + u32(gx)];
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
      var o = acc[j];
      if (p.useRelu == 1u) { o = max(o, 0.0); }
      outp[(coBase + j) * HW + y * p.W + x] = o;
    }
  }
}`;

const POOL_WGSL = /* wgsl */`
struct P { Ho:u32, Wo:u32, Hi:u32, Wi:u32, C:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       inp: array<f32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let c = gid.z;
  if (x >= p.Wo || y >= p.Ho || c >= p.C) { return; }
  let inBase = c * p.Hi * p.Wi;
  let y0 = y * 2u; let x0 = x * 2u;
  var m = inp[inBase + y0 * p.Wi + x0];
  m = max(m, inp[inBase + y0 * p.Wi + (x0 + 1u)]);
  m = max(m, inp[inBase + (y0 + 1u) * p.Wi + x0]);
  m = max(m, inp[inBase + (y0 + 1u) * p.Wi + (x0 + 1u)]);
  outp[c * p.Ho * p.Wo + y * p.Wo + x] = m;
}`;

const UP_WGSL = /* wgsl */`
struct P { Ho:u32, Wo:u32, Hi:u32, Wi:u32, C:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       inp: array<f32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let c = gid.z;
  if (x >= p.Wo || y >= p.Ho || c >= p.C) { return; }
  let iy = y / 2u; let ix = x / 2u;
  outp[c * p.Ho * p.Wo + y * p.Wo + x] = inp[c * p.Hi * p.Wi + iy * p.Wi + ix];
}`;

// sd003: GPU polygon rasterization to pre-reduce candidates, then exact CPU greedy.
//
// Greedy NMS is sequential (a polygon is suppressed only by an already-*kept* one),
// so it can't be reproduced exactly by one parallel pass — stamping all candidates
// into a single-owner grid fragments each true pairwise overlap across owners, so
// per-competitor overlaps read too low. But the GPU can cheaply do the part that
// dominates the cost: throwing out the ~30k near-duplicate proposals. A candidate
// whose area is almost entirely (>= `safe`, e.g. 90%) covered by ONE higher-scoring
// polygon is a redundant near-copy that greedy can never keep, so it's safe to drop.
// Fragmentation only makes that coverage read *lower*, so the pass never over-drops:
// the survivors are a superset of the true kept set, and the exact CPU greedy then
// finishes on that much smaller set — same result, a fraction of the rasterization.
//
// One thread per candidate scanline-fills its polygon. Pass `mode` 0 stamps a rank-
// owner grid (atomicMax(n-rank), so a pixel ends owned by its highest-scoring
// coverer and the grid clears to 0) and accumulates area. Pass 1 re-rasterizes,
// tallies how many of the candidate's pixels each distinct higher owner holds (a
// 16-slot per-thread table — enough for the neighbours a star polygon touches;
// overflow is skipped, which only reduces coverage further, staying safe), and
// clears the candidate's `keep` flag iff some single higher owner covers >= `safe`
// of it. `safe` arrives as parts-per-million in `thrM`.
const RASTER_WGSL = /* wgsl */`
const PI = 3.14159265358979;
struct P { n:u32, H:u32, W:u32, mode:u32, thrM:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read> cy: array<f32>;
@group(0) @binding(2) var<storage,read> cx: array<f32>;
@group(0) @binding(3) var<storage,read> dists: array<f32>;
@group(0) @binding(4) var<storage,read_write> owner: array<atomic<u32>>;
@group(0) @binding(5) var<storage,read_write> area: array<atomic<u32>>;
@group(0) @binding(6) var<storage,read_write> keep: array<atomic<u32>>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let i = gid.x;
  if (i >= p.n) { return; }
  let hi = p.n - i;                                     // owner grid stores n-rank (bigger = higher score)
  var vy: array<f32,32>;
  var vx: array<f32,32>;
  var ymin = 1e30; var ymax = -1e30;
  for (var k = 0u; k < 32u; k = k + 1u) {
    let dv = dists[i * 32u + k];
    let ang = 2.0 * PI * f32(k) / 32.0;
    let y = cy[i] + dv * sin(ang);
    let x = cx[i] + dv * cos(ang);
    vy[k] = y; vx[k] = x;
    ymin = min(ymin, y); ymax = max(ymax, y);
  }
  let W = p.W;
  var y0 = i32(ceil(ymin)); if (y0 < 0) { y0 = 0; }
  var y1 = i32(floor(ymax)); if (y1 > i32(p.H) - 1) { y1 = i32(p.H) - 1; }
  var oid: array<u32,16>;                              // mode 1: per-higher-owner overlap tally
  var ocnt: array<u32,16>;
  var nO = 0u;
  for (var y = y0; y <= y1; y = y + 1) {
    let fy = f32(y);
    var xi: array<f32,32>;
    var cnt = 0u;
    for (var k = 0u; k < 32u; k = k + 1u) {           // scanline / edge intersections
      let a = k; let b = (k + 1u) % 32u;
      let ya = vy[a]; let yb = vy[b];
      if ((ya <= fy && fy < yb) || (yb <= fy && fy < ya)) {
        xi[cnt] = vx[a] + (fy - ya) * (vx[b] - vx[a]) / (yb - ya);
        cnt = cnt + 1u;
      }
    }
    for (var a = 1u; a < cnt; a = a + 1u) {            // insertion sort xi[0..cnt)
      let key = xi[a]; var b = i32(a) - 1;
      loop {
        if (b >= 0 && xi[u32(b)] > key) { xi[u32(b) + 1u] = xi[u32(b)]; b = b - 1; }
        else { break; }
      }
      xi[u32(b) + 1u] = key;
    }
    var a = 0u;
    loop {
      if (a + 1u >= cnt) { break; }
      var xl = i32(ceil(xi[a])); if (xl < 0) { xl = 0; }
      var xr = i32(floor(xi[a + 1u])); if (xr > i32(W) - 1) { xr = i32(W) - 1; }
      for (var x = xl; x <= xr; x = x + 1) {
        let px = u32(y) * W + u32(x);
        if (p.mode == 0u) {
          atomicMax(&owner[px], hi);
          atomicAdd(&area[i], 1u);
        } else {
          let ov = atomicLoad(&owner[px]);
          if (ov > hi) {                                // owned by a strictly-higher polygon
            let o = p.n - ov;                           // that polygon's rank
            var f = -1;
            for (var t = 0u; t < nO; t = t + 1u) { if (oid[t] == o) { f = i32(t); break; } }
            if (f >= 0) { ocnt[u32(f)] = ocnt[u32(f)] + 1u; }
            else if (nO < 16u) { oid[nO] = o; ocnt[nO] = 1u; nO = nO + 1u; }
          }
        }
      }
      a = a + 2u;
    }
  }
  if (p.mode == 1u) {                                   // drop near-duplicates only (safe reduction)
    let safe = f32(p.thrM) * 1e-6;
    let ai = f32(atomicLoad(&area[i]));
    var maxov = 0u;
    for (var t = 0u; t < nO; t = t + 1u) { maxov = max(maxov, ocnt[t]); }
    if (f32(maxov) >= safe * ai) { atomicStore(&keep[i], 0u); }
  }
}`;

export class StarDistWebGPU {
  constructor(device) {
    this.device = device;
    const d = device;
    const mk = (code) => d.createComputePipeline({
      layout: "auto", compute: { module: d.createShaderModule({ code }), entryPoint: "main" }
    });
    this.pConv = mk(CONV_WGSL);
    this.pPool = mk(POOL_WGSL);
    this.pUp = mk(UP_WGSL);
    this.pRast = mk(RASTER_WGSL);
    this.buf = {};
    this._pool = new Map();
    this._inUse = [];
  }

  static async create() {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const lim = adapter.limits;
    const device = await adapter.requestDevice({
      requiredLimits: {
        maxBufferSize: lim.maxBufferSize,
        maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
        maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
      }
    });
    return new StarDistWebGPU(device);
  }

  loadWeights(manifest, binArrayBuffer) {
    this.manifest = manifest;
    this.tensors = manifest.tensors;
    this.blob = new Float32Array(binArrayBuffer);
    const d = this.device;
    const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    for (const name in this.tensors) {
      const t = this.tensors[name];
      const arr = this.blob.subarray(t.offset, t.offset + t.length);
      const buf = d.createBuffer({ size: Math.max(16, t.length * 4), usage: U });
      d.queue.writeBuffer(buf, 0, arr);
      this.buf[name] = buf;
    }
    // The first conv's input channels (1 for the grayscale fluo model, 3 for the
    // RGB H&E model) — read straight off its weight tensor rather than duplicated
    // as separate manifest metadata.
    this.nChannelIn = this.tensors["conv2d.w"].shape[1];
  }

  // Convenience loader: fetch manifest.json + weights.bin from a base URL and
  // return a ready instance. Default resolves to this repo's weights/stardist-fluo/
  // relative to this module; pass a CDN/HuggingFace base URL for production.
  static async load(baseURL = new URL("../weights/stardist-fluo/", import.meta.url).href,
                    { manifest = "manifest.json", weights = "weights.bin" } = {}) {
    const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
    const [mf, bin] = await Promise.all([
      fetch(base + manifest).then(r => r.json()),
      fetch(base + weights).then(r => r.arrayBuffer()),
    ]);
    const inst = await this.create();
    inst.loadWeights(mf, bin);
    return inst;
  }

  // ---- pooled scratch buffers (same free-list scheme as cellpose_core) ----
  _acquire(usage, sizeBytes) {
    const sz = Math.max(256, sizeBytes);
    const key = usage + ":" + sz;
    let free = this._pool.get(key);
    if (free && free.length) { const b = free.pop(); this._inUse.push([key, b]); return b; }
    const b = this.device.createBuffer({ size: sz, usage });
    if (!free) this._pool.set(key, []);
    this._inUse.push([key, b]);
    return b;
  }
  releaseAll() {
    for (const [key, b] of this._inUse) this._pool.get(key).push(b);
    this._inUse = [];
  }
  mkStorage(nfloats) {
    return this._acquire(
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, nfloats * 4);
  }
  uniform(ints) {
    const b = this._acquire(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, ints.length * 4);
    this.device.queue.writeBuffer(b, 0, new Uint32Array(ints));
    return b;
  }
  readback(sizeBytes) {
    return this._acquire(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, sizeBytes);
  }

  // ---- ops (record into encoder) ----
  conv(enc, name, inBuf, outBuf, H, W, Cin, Cout, K, relu) {
    const uni = this.uniform([H, W, Cin, Cout, K, (K / 2) | 0, relu ? 1 : 0, 0]);
    const bg = this.device.createBindGroup({
      layout: this.pConv.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: this.buf[name + ".w"] } },
        { binding: 3, resource: { buffer: this.buf[name + ".b"] } },
        { binding: 4, resource: { buffer: outBuf } },
      ]
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pConv); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16), Math.ceil(Cout / BLK));
    pass.end();
  }
  pool(enc, inBuf, outBuf, Hi, Wi, C) {
    const Ho = Hi >> 1, Wo = Wi >> 1;
    const uni = this.uniform([Ho, Wo, Hi, Wi, C]);
    const bg = this.device.createBindGroup({
      layout: this.pPool.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ]
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pPool); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Wo / 16), Math.ceil(Ho / 16), C);
    pass.end();
    return [Ho, Wo];
  }
  up(enc, inBuf, outBuf, Hi, Wi, C) {
    const Ho = Hi << 1, Wo = Wi << 1;
    const uni = this.uniform([Ho, Wo, Hi, Wi, C]);
    const bg = this.device.createBindGroup({
      layout: this.pUp.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ]
    });
    const pass = enc.beginComputePass();
    pass.setPipeline(this.pUp); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Wo / 16), Math.ceil(Ho / 16), C);
    pass.end();
    return [Ho, Wo];
  }
  // Concat along the channel axis. Feature maps are [C,H,W] and contiguous, so a
  // channel-axis concat is just the two flat buffers laid end to end — no shader.
  concat(enc, aBuf, Ca, bBuf, Cb, outBuf, HW) {
    enc.copyBufferToBuffer(aBuf, 0, outBuf, 0, Ca * HW * 4);
    enc.copyBufferToBuffer(bBuf, 0, outBuf, Ca * HW * 4, Cb * HW * 4);
  }

  // ---- U-Net forward: [Cin,Hp,Wp] input -> prob[gh,gw] + dist[gh,gw,32] ----
  // Cin is 1 for the grayscale fluo model, 3 for the RGB H&E model (this.nChannelIn,
  // set from the loaded weights' first conv shape).
  async forwardFromInput(inputF32, Hp, Wp) {
    const d = this.device;
    const inBuf = this.mkStorage(this.nChannelIn * Hp * Wp);
    d.queue.writeBuffer(inBuf, 0, inputF32);
    const enc = d.createCommandEncoder();
    const C = (name, i, o, H, W, cin, cout, k, relu) => {
      const b = this.mkStorage(cout * H * W);
      this.conv(enc, name, i, b, H, W, cin, cout, k, relu);
      return b;
    };
    // grid block (full res) -> pool to the (2,2) grid
    const c0 = C("conv2d", inBuf, null, Hp, Wp, this.nChannelIn, 32, 3, true);
    const c1 = C("conv2d_1", c0, null, Hp, Wp, 32, 32, 3, true);
    const gh = Hp >> 1, gw = Wp >> 1;
    const p0 = this.mkStorage(32 * gh * gw); this.pool(enc, c1, p0, Hp, Wp, 32);
    // encoder (all at grid res and below)
    const d0 = C("down_level_0_no_1", C("down_level_0_no_0", p0, null, gh, gw, 32, 32, 3, true),
      null, gh, gw, 32, 32, 3, true);
    const h1 = gh >> 1, w1 = gw >> 1;
    const p1 = this.mkStorage(32 * h1 * w1); this.pool(enc, d0, p1, gh, gw, 32);
    const d1 = C("down_level_1_no_1", C("down_level_1_no_0", p1, null, h1, w1, 32, 64, 3, true),
      null, h1, w1, 64, 64, 3, true);
    const h2 = h1 >> 1, w2 = w1 >> 1;
    const p2 = this.mkStorage(64 * h2 * w2); this.pool(enc, d1, p2, h1, w1, 64);
    const d2 = C("down_level_2_no_1", C("down_level_2_no_0", p2, null, h2, w2, 64, 128, 3, true),
      null, h2, w2, 128, 128, 3, true);
    const h3 = h2 >> 1, w3 = w2 >> 1;
    const p3 = this.mkStorage(128 * h3 * w3); this.pool(enc, d2, p3, h2, w2, 128);
    // bottleneck
    const m = C("middle_2", C("middle_0", p3, null, h3, w3, 128, 256, 3, true),
      null, h3, w3, 256, 128, 3, true);
    // decoder: upsample, concat encoder skip (up first, then skip), conv x2
    const u2 = this.mkStorage(128 * h2 * w2); this.up(enc, m, u2, h3, w3, 128);
    const cat2 = this.mkStorage(256 * h2 * w2); this.concat(enc, u2, 128, d2, 128, cat2, h2 * w2);
    const e2 = C("up_level_2_no_2", C("up_level_2_no_0", cat2, null, h2, w2, 256, 128, 3, true),
      null, h2, w2, 128, 64, 3, true);
    const u1 = this.mkStorage(64 * h1 * w1); this.up(enc, e2, u1, h2, w2, 64);
    const cat1 = this.mkStorage(128 * h1 * w1); this.concat(enc, u1, 64, d1, 64, cat1, h1 * w1);
    const e1 = C("up_level_1_no_2", C("up_level_1_no_0", cat1, null, h1, w1, 128, 64, 3, true),
      null, h1, w1, 64, 32, 3, true);
    const u0 = this.mkStorage(32 * gh * gw); this.up(enc, e1, u0, h1, w1, 32);
    const cat0 = this.mkStorage(64 * gh * gw); this.concat(enc, u0, 32, d0, 32, cat0, gh * gw);
    const e0 = C("up_level_0_no_2", C("up_level_0_no_0", cat0, null, gh, gw, 64, 32, 3, true),
      null, gh, gw, 32, 32, 3, true);
    // shared features + two heads (prob -> sigmoid in JS; dist -> linear)
    const feat = C("features", e0, null, gh, gw, 32, 128, 3, true);
    const probB = C("prob", feat, null, gh, gw, 128, 1, 1, false);
    const distB = C("dist", feat, null, gh, gw, 128, 32, 1, false);

    const probRB = this.readback(gh * gw * 4);
    const distRB = this.readback(N_RAYS * gh * gw * 4);
    enc.copyBufferToBuffer(probB, 0, probRB, 0, gh * gw * 4);
    enc.copyBufferToBuffer(distB, 0, distRB, 0, N_RAYS * gh * gw * 4);
    d.queue.submit([enc.finish()]);
    await Promise.all([probRB.mapAsync(GPUMapMode.READ), distRB.mapAsync(GPUMapMode.READ)]);
    const probRaw = new Float32Array(probRB.getMappedRange().slice(0));
    const distCHW = new Float32Array(distRB.getMappedRange().slice(0));  // [32,gh,gw]
    probRB.unmap(); distRB.unmap();
    this.releaseAll();

    // sigmoid the prob head; leave dist as [32,gh,gw] (the post-process indexes it)
    const prob = new Float32Array(gh * gw);
    for (let i = 0; i < prob.length; i++) prob[i] = 1 / (1 + Math.exp(-probRaw[i]));
    return { prob, dist: distCHW, gh, gw };
  }

  // Gather candidate polygons: grid pixels with prob>threshold (minus a `border`-px
  // frame), sorted by descending score, with their full-res centers + 32 distances.
  _candidates(prob, dist, gh, gw, opts) {
    const { prob_thresh = 0.4791, grid = [2, 2], border = 2 } = opts;
    const pts = [];
    for (let gy = border; gy < gh - border; gy++)
      for (let gx = border; gx < gw - border; gx++) {
        const s = prob[gy * gw + gx];
        if (s > prob_thresh) pts.push([gy, gx, s]);
      }
    pts.sort((a, b) => b[2] - a[2]);                 // descending score
    const n = pts.length;
    const cy = new Float64Array(n), cx = new Float64Array(n);
    const dists = new Float32Array(n * N_RAYS);
    const HWg = gh * gw;
    for (let i = 0; i < n; i++) {
      const [gy, gx] = pts[i];
      cy[i] = gy * grid[0]; cx[i] = gx * grid[1];
      for (let k = 0; k < N_RAYS; k++) dists[i * N_RAYS + k] = dist[k * HWg + gy * gw + gx];  // [32,gh,gw]
    }
    return { n, cy, cx, dists };
  }

  // ---- post-process: polygons + NMS on the CPU (JS port of mini_stardist) ----
  computeMasks(prob, dist, gh, gw, Hp, Wp, opts = {}) {
    const { nms_thresh = 0.3 } = opts;
    const { n, cy, cx, dists } = this._candidates(prob, dist, gh, gw, opts);
    if (!n) return new Int32Array(Hp * Wp);
    const kept = nmsPolygons(cy, cx, dists, n, Hp, Wp, nms_thresh);
    return polygonsToLabel(cy, cx, dists, kept, Hp, Wp);
  }

  // ---- sd003: GPU pre-reduces candidates, exact CPU greedy finishes ----
  // The GPU rasterizes all candidates (the CPU hotspot) and safely drops near-
  // duplicate proposals; the CPU greedy then runs on the small survivor set for the
  // exact same result. `safe` (default 0.9) is the coverage above which a candidate
  // is treated as a redundant near-copy of a higher one.
  async computeMasksGPU(prob, dist, gh, gw, Hp, Wp, opts = {}) {
    const { nms_thresh = 0.3, safe = 0.9 } = opts;
    const { n, cy, cx, dists } = this._candidates(prob, dist, gh, gw, opts);
    if (!n) return new Int32Array(Hp * Wp);
    const d = this.device;
    const cyB = this.mkStorage(n), cxB = this.mkStorage(n), distsB = this.mkStorage(n * N_RAYS);
    d.queue.writeBuffer(cyB, 0, Float32Array.from(cy));
    d.queue.writeBuffer(cxB, 0, Float32Array.from(cx));
    d.queue.writeBuffer(distsB, 0, dists);
    const ownerB = this.mkStorage(Hp * Wp), areaB = this.mkStorage(n), keepB = this.mkStorage(n);
    d.queue.writeBuffer(keepB, 0, new Uint32Array(n).fill(1));   // 1 = survivor (kept for the CPU pass)
    const thrM = Math.round(safe * 1e6);                        // safe-coverage in ppm (uniform is u32)

    const enc = d.createCommandEncoder();
    enc.clearBuffer(ownerB); enc.clearBuffer(areaB);            // owner encodes n-rank, so 0 = unowned
    for (const mode of [0, 1]) {                                // 0: stamp owner+area; 1: drop near-dups
      const uni = this.uniform([n, Hp, Wp, mode, thrM]);
      const bg = d.createBindGroup({
        layout: this.pRast.getBindGroupLayout(0), entries: [
          { binding: 0, resource: { buffer: uni } },
          { binding: 1, resource: { buffer: cyB } }, { binding: 2, resource: { buffer: cxB } },
          { binding: 3, resource: { buffer: distsB } }, { binding: 4, resource: { buffer: ownerB } },
          { binding: 5, resource: { buffer: areaB } }, { binding: 6, resource: { buffer: keepB } },
        ]
      });
      const pass = enc.beginComputePass();
      pass.setPipeline(this.pRast); pass.setBindGroup(0, bg);
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
    }
    const keepRB = this.readback(n * 4);
    enc.copyBufferToBuffer(keepB, 0, keepRB, 0, n * 4);
    d.queue.submit([enc.finish()]);
    await keepRB.mapAsync(GPUMapMode.READ);
    const keep = new Uint32Array(keepRB.getMappedRange().slice(0));
    keepRB.unmap();
    this.releaseAll();

    // survivors, still in descending-score order; exact greedy NMS on just these
    let m = 0;
    const cyS = new Float64Array(n), cxS = new Float64Array(n), distsS = new Float32Array(n * N_RAYS);
    for (let i = 0; i < n; i++) {
      if (keep[i] !== 1) continue;
      cyS[m] = cy[i]; cxS[m] = cx[i];
      distsS.set(dists.subarray(i * N_RAYS, i * N_RAYS + N_RAYS), m * N_RAYS);
      m++;
    }
    const kept = nmsPolygons(cyS, cxS, distsS, m, Hp, Wp, nms_thresh);
    return polygonsToLabel(cyS, cxS, distsS, kept, Hp, Wp);
  }

  // `input` is a single grayscale [H,W] Float32Array (fluo, Cin=1) or an array of
  // `nChannelIn` such arrays (H&E, Cin=3, e.g. [r,g,b]) — each channel is percentile-
  // normalized independently (StarDist's own axis_norm=(0,1) convention) before
  // being reflect-padded and stacked into the net's [Cin,Hp,Wp] input.
  async segmentImage(input, H, W, opts = {}) {
    const { gpu = true } = opts;
    const t0 = now();
    const channels = Array.isArray(input) ? input : [input];
    const normed = channels.map(ch => normalize99(ch));
    const { data, Hp, Wp } = padChannelsTo16(normed, H, W);
    const t1 = now();
    const { prob, dist, gh, gw } = await this.forwardFromInput(data, Hp, Wp);
    const t2 = now();
    const full = gpu ? await this.computeMasksGPU(prob, dist, gh, gw, Hp, Wp, opts)
                     : this.computeMasks(prob, dist, gh, gw, Hp, Wp, opts);
    // crop the [Hp,Wp] label image back to [H,W]
    const labels = new Int32Array(H * W);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) labels[y * W + x] = full[y * Wp + x];
    const t3 = now();
    return { labels, timings: { preprocess: t1 - t0, forward: t2 - t1, nms: t3 - t2, total: t3 - t0 } };
  }
}

// ======================================================================
// Geometry + NMS (JS port of mini_stardist.py) — module functions
// ======================================================================
const RAY_SIN = new Float64Array(N_RAYS), RAY_COS = new Float64Array(N_RAYS);
for (let k = 0; k < N_RAYS; k++) {
  const phi = 2 * Math.PI * k / N_RAYS;           // linspace(0,2pi,32,endpoint=false)
  RAY_SIN[k] = Math.sin(phi); RAY_COS[k] = Math.cos(phi);
}

// Rasterize one star-convex polygon (center cy,cx + 32 distances) into `out`,
// filling out[0..count) with the flat indices y*W+x of the pixels inside it and
// returning count. Even-odd scanline fill sampling pixel centres — the
// inside/outside rule skimage.draw.polygon uses. `out` is a caller-owned scratch
// buffer reused across polygons, so there's no per-polygon allocation (and, vs
// the earlier Set-of-indices, no hashing) — the hot path for the dense image.
function rasterPolygon(cy, cx, dOff, dists, H, W, out) {
  const ys = new Float64Array(N_RAYS), xs = new Float64Array(N_RAYS);
  let ymin = Infinity, ymax = -Infinity;
  for (let k = 0; k < N_RAYS; k++) {
    const dv = dists[dOff + k];
    const y = cy + dv * RAY_SIN[k], x = cx + dv * RAY_COS[k];
    ys[k] = y; xs[k] = x;
    if (y < ymin) ymin = y; if (y > ymax) ymax = y;
  }
  const y0 = Math.max(0, Math.ceil(ymin)), y1 = Math.min(H - 1, Math.floor(ymax));
  const xints = [];
  let cnt = 0;
  for (let y = y0; y <= y1; y++) {
    xints.length = 0;
    for (let k = 0; k < N_RAYS; k++) {
      const ya = ys[k], xa = xs[k];
      const kn = (k + 1) % N_RAYS, yb = ys[kn], xb = xs[kn];
      if ((ya <= y && y < yb) || (yb <= y && y < ya)) {
        xints.push(xa + (y - ya) * (xb - xa) / (yb - ya));
      }
    }
    xints.sort((a, b) => a - b);
    for (let a = 0; a + 1 < xints.length; a += 2) {
      const xl = Math.max(0, Math.ceil(xints[a])), xr = Math.min(W - 1, Math.floor(xints[a + 1]));
      const rowBase = y * W;
      for (let x = xl; x <= xr; x++) out[cnt++] = rowBase + x;
    }
  }
  return cnt;
}

// Greedy NMS, StarDist's rule: walking candidates from highest score down, a
// polygon is suppressed by an already-kept one if their overlap Ainter/min(A1,A2)
// exceeds nms_thresh (min-area overlap, not union IoU). Inputs are score-descending.
//
// sd002: instead of storing each kept polygon's pixels and intersecting sets, we
// stamp kept polygons into a single `owner` grid (each pixel holds the id of the
// first — i.e. highest-scoring — kept polygon covering it). A candidate is then
// rasterized once, and one linear pass tallies how many of its pixels each owner
// already holds, giving every pairwise overlap at once with no per-kept loop and
// no Set hashing. Because stamping is first-come (a pixel keeps its highest-score
// owner), an owner's tally slightly under-counts where two kept polygons overlap;
// kept polygons only ever overlap by ≤ nms_thresh, so the effect is tiny and the
// end-to-end AP is unchanged (see test_stardist.js).
function nmsPolygons(cy, cx, dists, n, H, W, nms_thresh) {
  const owner = new Int32Array(H * W);   // 0 = free, else 1-based kept id
  const keptArea = [0];                  // area per kept id (keptArea[id])
  const kept = [];
  const scratch = new Int32Array(H * W);
  const counts = new Map();
  for (let i = 0; i < n; i++) {
    const area = rasterPolygon(cy[i], cx[i], i * N_RAYS, dists, H, W, scratch);
    if (area === 0) continue;
    counts.clear();
    for (let t = 0; t < area; t++) {
      const o = owner[scratch[t]];
      if (o !== 0) counts.set(o, (counts.get(o) || 0) + 1);
    }
    let suppressed = false;
    for (const [o, c] of counts) {
      if (c / Math.min(area, keptArea[o]) > nms_thresh) { suppressed = true; break; }
    }
    if (!suppressed) {
      const id = kept.length + 1;
      kept.push(i); keptArea.push(area);
      for (let t = 0; t < area; t++) { const px = scratch[t]; if (owner[px] === 0) owner[px] = id; }
    }
  }
  return kept;
}

// Rasterize the kept polygons into a label image, lowest score first so the
// highest-probability polygon is painted last and wins overlaps (the arrays are
// score-descending, so we walk `kept` in reverse).
function polygonsToLabel(cy, cx, dists, kept, H, W) {
  const lbl = new Int32Array(H * W);
  const scratch = new Int32Array(H * W);
  let lab = 1;
  for (let m = kept.length - 1; m >= 0; m--) {
    const i = kept[m];
    const area = rasterPolygon(cy[i], cx[i], i * N_RAYS, dists, H, W, scratch);
    for (let t = 0; t < area; t++) lbl[scratch[t]] = lab;
    lab++;
  }
  return lbl;
}

// ======================================================================
// Preprocessing (single grayscale channel)
// ======================================================================
// Percentile contrast-stretch to the 1st..99.8th percentile (csbdeep/StarDist
// default). Guards a constant image (leaves it) and a dead-range one (zeros it),
// mirroring csbdeep — harmless for real input, safe for a blank channel.
export function normalize99(x, lower = 1, upper = 99.8) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < x.length; i++) { const v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const out = new Float32Array(x.length);
  if (!(mx > mn)) { out.set(x); return out; }
  const s = Float32Array.from(x).sort();
  const p = (q) => s[Math.min(s.length - 1, Math.max(0, Math.floor(q / 100 * (s.length - 1))))];
  const p1 = p(lower), p99 = p(upper);
  if (p99 - p1 <= 1e-3) return out;
  const denom = (p99 - p1) + 1e-20;
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - p1) / denom;
  return out;
}

// Reflect-pad a single [H,W] channel to a multiple of 16 (the net pools 4×),
// returning the [1,Hp,Wp] net input. Matches export_stardist.pad_to.
export function padTo16(ch, H, W) {
  const div = 16;
  const Hp = Math.ceil(H / div) * div, Wp = Math.ceil(W / div) * div;
  const data = new Float32Array(Hp * Wp);
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data[y * Wp + x] = ch[y * W + x];
  // Symmetric reflection with the edge included (matches export_stardist.pad_to's
  // `x[H-1::-1]`): first pad row = row H-1, next = H-2, ... (NOT edge-excluded).
  for (let y = H; y < Hp; y++) for (let x = 0; x < W; x++)                 // reflect rows
    data[y * Wp + x] = data[Math.max(0, 2 * H - 1 - y) * Wp + x];
  for (let y = 0; y < Hp; y++) for (let x = W; x < Wp; x++)                 // reflect cols
    data[y * Wp + x] = data[y * Wp + Math.max(0, 2 * W - 1 - x)];
  return { data, Hp, Wp };
}

// Reflect-pad each of `channels` (already-normalized [H,W] arrays) the same way as
// padTo16, then stack them into a single [Cin,Hp,Wp] net input.
export function padChannelsTo16(channels, H, W) {
  const Cin = channels.length;
  const { Hp, Wp } = padTo16(channels[0], H, W);
  const data = new Float32Array(Cin * Hp * Wp);
  channels.forEach((ch, c) => {
    const { data: chData } = padTo16(ch, H, W);
    data.set(chData, c * Hp * Wp);
  });
  return { data, Hp, Wp };
}

function now() {
  return (typeof performance !== "undefined" ? performance.now() : Date.now());
}
