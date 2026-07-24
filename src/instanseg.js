// InstanSeg (brightfield_nuclei) inference on WebGPU — environment-agnostic core.
// Third sibling to cellpose_core.js / stardist_core.js. WGSL for the InstanSeg_UNet
// forward; the learned instance decode (seed peaks + per-seed classifier MLP +
// flood-fill + IoU merge) runs in JS.
//
// Public API:
//   const is = await InstanSegWebGPU.create();
//   is.loadWeights(manifestObj, binArrayBuffer);
//   const {out, H, W} = await is.forwardFromInput(inputF32, Hp, Wp);   // [5,Hp,Wp]
//   const {labels, timings} = await is.segmentImage(rgb, H, W, opts);  // rgb: [3,H,W]
//
// The net is post-activation conv→BN→ReLU with BN already folded into each conv by
// export_instanseg.py, so the conv shader is just conv (+ optional ReLU). The
// residual/skip merges are elementwise adds (ADD_WGSL). Output = 2 coord fields +
// 2 sigma + 1 seed channel; the decode grows one instance from each seed.

const BLK = 8, TS = 16, TW = TS + 2;
const N_COORD = 2, N_SIGMA = 2;

const CONV_WGSL = /* wgsl */`
const BLK = ${BLK}u; const TS = ${TS}u; const TW = ${TW}u;
struct P { H:u32, W:u32, Cin:u32, Cout:u32, K:u32, pad:u32, useRelu:u32, _p:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       inp:  array<f32>;
@group(0) @binding(2) var<storage,read>       w:    array<f32>;
@group(0) @binding(3) var<storage,read>       b:    array<f32>;
@group(0) @binding(4) var<storage,read_write> outp: array<f32>;
var<workgroup> tile: array<f32, TW * TW>;
var<workgroup> ws:   array<f32, BLK * 9u>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {
  let coBase = wg.z * BLK; let nco = min(BLK, p.Cout - coBase); let HW = p.H * p.W;
  let K = p.K; let pad = i32(p.pad); let KK = K * K;
  let x = wg.x * TS + lid.x; let y = wg.y * TS + lid.y; let lt = lid.y * TS + lid.x;
  let ox = i32(wg.x * TS) - pad; let oy = i32(wg.y * TS) - pad;
  var acc: array<f32, BLK>;
  for (var j = 0u; j < BLK; j = j + 1u) { acc[j] = b[coBase + min(j, nco - 1u)]; }
  let tileN = TW * TW; let wN = nco * KK; let stride = p.Cin * KK;
  for (var ci = 0u; ci < p.Cin; ci = ci + 1u) {
    let base = ci * HW;
    for (var i = lt; i < tileN; i = i + TS * TS) {
      let ty = i / TW; let tx = i % TW; let gy = oy + i32(ty); let gx = ox + i32(tx);
      var v = 0.0;
      if (gy >= 0 && gy < i32(p.H) && gx >= 0 && gx < i32(p.W)) { v = inp[base + u32(gy) * p.W + u32(gx)]; }
      tile[i] = v;
    }
    let wbase = ci * KK;
    for (var i = lt; i < wN; i = i + TS * TS) { let j = i / KK; let k = i % KK; ws[i] = w[(coBase + j) * stride + wbase + k]; }
    workgroupBarrier();
    for (var ky = 0u; ky < K; ky = ky + 1u) {
      for (var kx = 0u; kx < K; kx = kx + 1u) {
        let v = tile[(lid.y + ky) * TW + (lid.x + kx)]; let k = ky * K + kx;
        for (var j = 0u; j < nco; j = j + 1u) { acc[j] = acc[j] + v * ws[j * KK + k]; }
      }
    }
    workgroupBarrier();
  }
  if (x < p.W && y < p.H) {
    for (var j = 0u; j < nco; j = j + 1u) {
      var o = acc[j]; if (p.useRelu == 1u) { o = max(o, 0.0); }
      outp[(coBase + j) * HW + y * p.W + x] = o;
    }
  }
}`;

const POOL_WGSL = /* wgsl */`
struct P { Ho:u32, Wo:u32, Hi:u32, Wi:u32, C:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read> inp: array<f32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let c = gid.z;
  if (x >= p.Wo || y >= p.Ho || c >= p.C) { return; }
  let ib = c * p.Hi * p.Wi; let y0 = y * 2u; let x0 = x * 2u;
  var m = inp[ib + y0 * p.Wi + x0];
  m = max(m, inp[ib + y0 * p.Wi + (x0 + 1u)]);
  m = max(m, inp[ib + (y0 + 1u) * p.Wi + x0]);
  m = max(m, inp[ib + (y0 + 1u) * p.Wi + (x0 + 1u)]);
  outp[c * p.Ho * p.Wo + y * p.Wo + x] = m;
}`;

const UP_WGSL = /* wgsl */`
struct P { Ho:u32, Wo:u32, Hi:u32, Wi:u32, C:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read> inp: array<f32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let x = gid.x; let y = gid.y; let c = gid.z;
  if (x >= p.Wo || y >= p.Ho || c >= p.C) { return; }
  outp[c * p.Ho * p.Wo + y * p.Wo + x] = inp[c * p.Hi * p.Wi + (y / 2u) * p.Wi + (x / 2u)];
}`;

const ADD_WGSL = /* wgsl */`
struct P { N:u32, nwgx:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read> a: array<f32>;
@group(0) @binding(2) var<storage,read> b: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let i = (wg.y * p.nwgx + wg.x) * 256u + li.x;
  if (i >= p.N) { return; }
  o[i] = a[i] + b[i];
}`;

export class InstanSegWebGPU {
  constructor(device) {
    this.device = device;
    const mk = (code) => device.createComputePipeline({
      layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    this.pConv = mk(CONV_WGSL); this.pPool = mk(POOL_WGSL); this.pUp = mk(UP_WGSL); this.pAdd = mk(ADD_WGSL);
    this.buf = {}; this._pool = new Map(); this._inUse = [];
  }
  static async create() {
    const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
    if (!adapter) throw new Error("no WebGPU adapter");
    const lim = adapter.limits;
    const device = await adapter.requestDevice({ requiredLimits: {
      maxBufferSize: lim.maxBufferSize, maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup } });
    return new InstanSegWebGPU(device);
  }
  loadWeights(manifest, binArrayBuffer) {
    this.manifest = manifest; this.tensors = manifest.tensors; this.relu = manifest.relu;
    this.blob = new Float32Array(binArrayBuffer);
    const d = this.device, U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST;
    this.K = {}; this.shape = {};
    for (const name in this.tensors) {
      const t = this.tensors[name];
      const buf = d.createBuffer({ size: Math.max(16, t.length * 4), usage: U });
      d.queue.writeBuffer(buf, 0, this.blob.subarray(t.offset, t.offset + t.length));
      this.buf[name] = buf; this.shape[name] = t.shape;
      if (name.endsWith(".w")) this.K[name.slice(0, -2)] = t.shape[3];   // kernel size per conv unit
    }
  }

  // Convenience loader: fetch manifest.json + weights.bin from a base URL and
  // return a ready instance. Default resolves to this repo's
  // weights/instanseg-brightfield/ relative to this module; pass a CDN/HuggingFace
  // base URL for production.
  static async load(baseURL = new URL("../weights/instanseg-brightfield/", import.meta.url).href,
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

  arr(name) { const t = this.tensors[name]; return this.blob.subarray(t.offset, t.offset + t.length); }

  // ---- pooled scratch (same scheme as the other cores) ----
  _acquire(usage, sz) {
    sz = Math.max(256, sz); const key = usage + ":" + sz; let free = this._pool.get(key);
    if (free && free.length) { const b = free.pop(); this._inUse.push([key, b]); return b; }
    const b = this.device.createBuffer({ size: sz, usage }); if (!free) this._pool.set(key, []);
    this._inUse.push([key, b]); return b;
  }
  releaseAll() { for (const [k, b] of this._inUse) this._pool.get(k).push(b); this._inUse = []; }
  mkStorage(n) { return this._acquire(GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST, n * 4); }
  uniform(ints) { const b = this._acquire(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, ints.length * 4);
    this.device.queue.writeBuffer(b, 0, new Uint32Array(ints)); return b; }
  readback(sz) { return this._acquire(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, sz); }

  // ---- ops ----
  conv(enc, tag, inBuf, outBuf, H, W, Cin, Cout, relu) {
    const K = this.K[tag];
    const uni = this.uniform([H, W, Cin, Cout, K, (K / 2) | 0, relu ? 1 : 0, 0]);
    const bg = this.device.createBindGroup({ layout: this.pConv.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: inBuf } },
      { binding: 2, resource: { buffer: this.buf[tag + ".w"] } }, { binding: 3, resource: { buffer: this.buf[tag + ".b"] } },
      { binding: 4, resource: { buffer: outBuf } } ] });
    const pass = enc.beginComputePass(); pass.setPipeline(this.pConv); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(W / 16), Math.ceil(H / 16), Math.ceil(Cout / BLK)); pass.end();
  }
  pool(enc, inBuf, outBuf, Hi, Wi, C) {
    const Ho = Hi >> 1, Wo = Wi >> 1; const uni = this.uniform([Ho, Wo, Hi, Wi, C]);
    const bg = this.device.createBindGroup({ layout: this.pPool.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: inBuf } }, { binding: 2, resource: { buffer: outBuf } } ] });
    const pass = enc.beginComputePass(); pass.setPipeline(this.pPool); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Wo / 16), Math.ceil(Ho / 16), C); pass.end(); return [Ho, Wo];
  }
  up(enc, inBuf, outBuf, Hi, Wi, C) {
    const Ho = Hi << 1, Wo = Wi << 1; const uni = this.uniform([Ho, Wo, Hi, Wi, C]);
    const bg = this.device.createBindGroup({ layout: this.pUp.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: inBuf } }, { binding: 2, resource: { buffer: outBuf } } ] });
    const pass = enc.beginComputePass(); pass.setPipeline(this.pUp); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Wo / 16), Math.ceil(Ho / 16), C); pass.end(); return [Ho, Wo];
  }
  add(enc, aBuf, bBuf, outBuf, N) {
    const nwg = Math.ceil(N / 256), nwgx = Math.min(nwg, 65535), nwgy = Math.ceil(nwg / nwgx);
    const uni = this.uniform([N, nwgx]);
    const bg = this.device.createBindGroup({ layout: this.pAdd.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: aBuf } },
      { binding: 2, resource: { buffer: bBuf } }, { binding: 3, resource: { buffer: outBuf } } ] });
    const pass = enc.beginComputePass(); pass.setPipeline(this.pAdd); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(nwgx, nwgy); pass.end();
  }

  // ---- InstanSeg_UNet forward: [3,Hp,Wp] -> [5,Hp,Wp] ----
  encBlock(enc, n, x, Cin, Cout, H, W, pool) {
    let src = x, h = H, w = W;
    if (pool) { const pb = this.mkStorage(Cin * (H >> 1) * (W >> 1)); [h, w] = this.pool(enc, x, pb, H, W, Cin); src = pb; }
    const N = Cout * h * w;
    const proj = this.mkStorage(N); this.conv(enc, `enc${n}.c0`, src, proj, h, w, Cin, Cout, true);
    const c1 = this.mkStorage(N); this.conv(enc, `enc${n}.c1`, src, c1, h, w, Cin, Cout, true);
    const c2 = this.mkStorage(N); this.conv(enc, `enc${n}.c2`, c1, c2, h, w, Cout, Cout, true);
    const x1 = this.mkStorage(N); this.add(enc, proj, c2, x1, N);              // proj + conv2(conv1)
    const c3 = this.mkStorage(N); this.conv(enc, `enc${n}.c3`, x1, c3, h, w, Cout, Cout, true);
    const c4 = this.mkStorage(N); this.conv(enc, `enc${n}.c4`, c3, c4, h, w, Cout, Cout, true);
    const out = this.mkStorage(N); this.add(enc, x1, c4, out, N);              // x + conv4(conv3)
    return { buf: out, h, w };
  }
  decBlock(enc, n, x, skip, Cin, Cout, Hi, Wi) {
    const upb = this.mkStorage(Cin * (Hi << 1) * (Wi << 1)); const [h, w] = this.up(enc, x, upb, Hi, Wi, Cin);
    const N = Cout * h * w;
    const proj = this.mkStorage(N); this.conv(enc, `dec${n}.c0`, upb, proj, h, w, Cin, Cout, true);
    const c1 = this.mkStorage(N); this.conv(enc, `dec${n}.c1`, upb, c1, h, w, Cin, Cout, true);
    const sk = this.mkStorage(N); this.conv(enc, `dec${n}.skip`, skip, sk, h, w, Cout, Cout, true);
    const merged = this.mkStorage(N); this.add(enc, c1, sk, merged, N);        // conv1 + conv_skip(skip)
    const c2 = this.mkStorage(N); this.conv(enc, `dec${n}.c2`, merged, c2, h, w, Cout, Cout, true);
    const x1 = this.mkStorage(N); this.add(enc, proj, c2, x1, N);              // proj + conv2(...)
    const c3 = this.mkStorage(N); this.conv(enc, `dec${n}.c3`, x1, c3, h, w, Cout, Cout, true);
    const c4 = this.mkStorage(N); this.conv(enc, `dec${n}.c4`, c3, c4, h, w, Cout, Cout, true);
    const out = this.mkStorage(N); this.add(enc, x1, c4, out, N);              // x + conv4(conv3)
    return { buf: out, h, w };
  }
  async forwardFromInput(inputF32, Hp, Wp) {
    const d = this.device;
    const inBuf = this.mkStorage(3 * Hp * Wp); d.queue.writeBuffer(inBuf, 0, inputF32);
    const enc = d.createCommandEncoder();
    const e0 = this.encBlock(enc, 0, inBuf, 3, 32, Hp, Wp, false);
    const e1 = this.encBlock(enc, 1, e0.buf, 32, 64, e0.h, e0.w, true);
    const e2 = this.encBlock(enc, 2, e1.buf, 64, 128, e1.h, e1.w, true);
    const e3 = this.encBlock(enc, 3, e2.buf, 128, 256, e2.h, e2.w, true);
    const d0 = this.decBlock(enc, 0, e3.buf, e2.buf, 256, 128, e3.h, e3.w);
    const d1 = this.decBlock(enc, 1, d0.buf, e1.buf, 128, 64, d0.h, d0.w);
    const d2 = this.decBlock(enc, 2, d1.buf, e0.buf, 64, 32, d1.h, d1.w);
    // three 1x1 heads -> [2,2,1], written into a single [5,Hp,Wp] buffer
    const HW = Hp * Wp, outB = this.mkStorage(5 * HW);
    const h0 = this.mkStorage(2 * HW); this.conv(enc, "head0", d2.buf, h0, Hp, Wp, 32, 2, false);
    const h1 = this.mkStorage(2 * HW); this.conv(enc, "head1", d2.buf, h1, Hp, Wp, 32, 2, false);
    const h2 = this.mkStorage(1 * HW); this.conv(enc, "head2", d2.buf, h2, Hp, Wp, 32, 1, false);
    enc.copyBufferToBuffer(h0, 0, outB, 0, 2 * HW * 4);
    enc.copyBufferToBuffer(h1, 0, outB, 2 * HW * 4, 2 * HW * 4);
    enc.copyBufferToBuffer(h2, 0, outB, 4 * HW * 4, 1 * HW * 4);
    const rb = this.readback(5 * HW * 4); enc.copyBufferToBuffer(outB, 0, rb, 0, 5 * HW * 4);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const out = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
    this.releaseAll();
    return { out, H: Hp, W: Wp };
  }

  // ---- decode (JS port of mini_instanseg) ----
  computeMasks(out, Hp, Wp, H, W, opts = {}) {
    const P = { ...DEFAULTS, ...opts };
    const HW = Hp * Wp;
    // fields = (sigmoid(out[0:2])-0.5)*8 + coordinate_map ; sigma = out[2:4] ; seed = out[4]/15+0.5
    const fx = new Float32Array(H * W), fy = new Float32Array(H * W);
    const s0 = new Float32Array(H * W), s1 = new Float32Array(H * W), seed = new Float32Array(H * W);
    // coordinate_map: linspace(0, dim*64/256, dim) endpoint-inclusive -> step dim*64/256/(dim-1)
    const cscaleX = (W * 64 / 256) / Math.max(1, W - 1), cscaleY = (H * 64 / 256) / Math.max(1, H - 1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, o = y * Wp + x;
      fx[i] = (sig(out[o]) - 0.5) * 8 + x * cscaleX;
      fy[i] = (sig(out[HW + o]) - 0.5) * 8 + y * cscaleY;
      s0[i] = out[2 * HW + o]; s1[i] = out[3 * HW + o];
      seed[i] = out[4 * HW + o] / 15 + 0.5;
    }
    const centroids = peakLocalMax(seed, H, W, P.peak_distance, P.seed_threshold);
    const C = centroids.length;
    if (C === 0) return new Int32Array(H * W);

    const sz = Math.min(P.window_size, H, W) * 2, half = sz >> 1;
    const bestProb = new Float32Array(H * W), labels = new Int32Array(H * W);
    const masks = [];
    const fc1w = this.arr("pc.fc1.w"), fc1b = this.arr("pc.fc1.b");
    const fc2w = this.arr("pc.fc2.w"), fc2b = this.arr("pc.fc2.b");
    const fc3w = this.arr("pc.fc3.w"), fc3b = this.arr("pc.fc3.b");
    for (let s = 0; s < C; s++) {
      const [cy, cx] = centroids[s];
      const ci = cy * W + cx, ccx = fx[ci], ccy = fy[ci];
      const y0 = clampi(cy, half, H - half) - half, x0 = clampi(cx, half, W - half) - half;
      const prob = new Float32Array(sz * sz);
      for (let a = 0; a < sz; a++) {
        const yy = clampi(y0 + a, 0, H - 1);
        for (let b = 0; b < sz; b++) {
          const xx = clampi(x0 + b, 0, W - 1), i = yy * W + xx;
          const f0 = fx[i] - ccx, f1 = fy[i] - ccy;
          // MLP 4->5->5->1
          let h0 = 0, h1 = 0, h2 = 0, h3 = 0, h4 = 0;
          const inp = [f0, f1, s0[i], s1[i]];
          const H1 = new Float32Array(5);
          for (let k = 0; k < 5; k++) { let v = fc1b[k]; for (let j = 0; j < 4; j++) v += fc1w[k * 4 + j] * inp[j]; H1[k] = Math.max(0, v); }
          const H2 = new Float32Array(5);
          for (let k = 0; k < 5; k++) { let v = fc2b[k]; for (let j = 0; j < 5; j++) v += fc2w[k * 5 + j] * H1[j]; H2[k] = Math.max(0, v); }
          let v = fc3b[0]; for (let j = 0; j < 5; j++) v += fc3w[j] * H2[j];
          prob[a * sz + b] = sig(v);
        }
      }
      let binary = new Uint8Array(sz * sz); for (let k = 0; k < sz * sz; k++) binary[k] = prob[k] >= P.mask_threshold ? 1 : 0;
      if (P.cleanup_fragments) {
        const syc = clampi(cy - y0, 0, sz - 1), sxc = clampi(cx - x0, 0, sz - 1);
        const keep = floodFillKeep(binary, sz, sz, syc, sxc);
        for (let k = 0; k < sz * sz; k++) { if (keep[k]) { if (prob[k] < P.mask_threshold) prob[k] = P.mask_threshold; } else prob[k] = 0; binary[k] = prob[k] >= P.mask_threshold ? 1 : 0; }
      }
      const mset = new Set();
      for (let a = 0; a < sz; a++) { const yy = clampi(y0 + a, 0, H - 1); for (let b = 0; b < sz; b++) {
        if (!binary[a * sz + b]) continue;
        const xx = clampi(x0 + b, 0, W - 1), i = yy * W + xx, pr = prob[a * sz + b];
        mset.add(i);
        if (pr > bestProb[i]) { bestProb[i] = pr; labels[i] = s + 1; }
      } }
      masks.push(mset);
    }
    return mergeAndFilter(labels, masks, C, P);
  }

  async segmentImage(rgb, H, W, opts = {}) {
    const t0 = now();
    const norm = percentileNormalize(rgb, H, W);
    const { data, Hp, Wp } = padTo8(norm, H, W);
    for (let i = 0; i < data.length; i++) data[i] = Math.min(3, Math.max(-2, data[i]));
    const t1 = now();
    const { out } = await this.forwardFromInput(data, Hp, Wp);
    const t2 = now();
    const labels = this.computeMasks(out, Hp, Wp, H, W, opts);
    const t3 = now();
    return { labels, timings: { preprocess: t1 - t0, forward: t2 - t1, decode: t3 - t2, total: t3 - t0 } };
  }
}

// ======================================================================
// decode helpers (JS port of mini_instanseg)
// ======================================================================
const DEFAULTS = { min_size: 10, mask_threshold: 0.53, peak_distance: 5, seed_threshold: 0.7,
  overlap_threshold: 0.3, window_size: 32, cleanup_fragments: true };
const sig = (x) => 1 / (1 + Math.exp(-x));
const clampi = (v, lo, hi) => v < lo ? lo : (v > hi ? hi : v);

function peakLocalMax(img, H, W, k, minimum) {
  // a pixel that equals the max of its (2k+1)^2 window and exceeds `minimum`
  const out = [];
  for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
    const v = img[y * W + x];
    if (v <= minimum) continue;
    let isMax = true;
    for (let dy = -k; dy <= k && isMax; dy++) {
      const yy = y + dy; if (yy < 0 || yy >= H) continue;
      for (let dx = -k; dx <= k; dx++) {
        const xx = x + dx; if (xx < 0 || xx >= W) continue;
        if (img[yy * W + xx] > v) { isMax = false; break; }
      }
    }
    if (isMax) out.push([y, x]);
  }
  return out;
}

function floodFillKeep(binary, H, W, sy, sx) {
  const keep = new Uint8Array(H * W);
  if (!binary[sy * W + sx]) { keep[sy * W + sx] = 1; return keep; }
  const st = [sy * W + sx]; keep[sy * W + sx] = 1;
  while (st.length) { const p = st.pop(), y = (p / W) | 0, x = p % W;
    const nb = [[y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]];
    for (const [ny, nx] of nb) if (ny >= 0 && ny < H && nx >= 0 && nx < W) { const q = ny * W + nx; if (binary[q] && !keep[q]) { keep[q] = 1; st.push(q); } }
  }
  // fill holes: background not reachable from the border
  const reach = new Uint8Array(H * W), st2 = [];
  for (let y = 0; y < H; y++) { for (const x of [0, W - 1]) { const q = y * W + x; if (!keep[q] && !reach[q]) { reach[q] = 1; st2.push(q); } } }
  for (let x = 0; x < W; x++) { for (const y of [0, H - 1]) { const q = y * W + x; if (!keep[q] && !reach[q]) { reach[q] = 1; st2.push(q); } } }
  while (st2.length) { const p = st2.pop(), y = (p / W) | 0, x = p % W;
    const nb = [[y - 1, x], [y + 1, x], [y, x - 1], [y, x + 1]];
    for (const [ny, nx] of nb) if (ny >= 0 && ny < H && nx >= 0 && nx < W) { const q = ny * W + nx; if (!keep[q] && !reach[q]) { reach[q] = 1; st2.push(q); } }
  }
  for (let k = 0; k < H * W; k++) if (!keep[k] && !reach[k]) keep[k] = 1;   // interior holes -> filled
  return keep;
}

function mergeAndFilter(labels, masks, C, P) {
  // union-find over seeds whose masks overlap (IoU > overlap_threshold)
  const parent = new Int32Array(C); for (let i = 0; i < C; i++) parent[i] = i;
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const areas = masks.map(m => m.size);
  for (let i = 0; i < C; i++) { if (!areas[i]) continue;
    for (let j = i + 1; j < C; j++) { if (!areas[j]) continue;
      const [small, big] = areas[i] < areas[j] ? [masks[i], masks[j]] : [masks[j], masks[i]];
      let inter = 0; for (const p of small) if (big.has(p)) inter++;
      if (inter && inter / (areas[i] + areas[j] - inter) > P.overlap_threshold) parent[find(i)] = find(j);
    }
  }
  const remap = new Int32Array(C + 1);
  for (let i = 0; i < C; i++) remap[i + 1] = find(i) + 1;                    // label -> component
  const merged = new Int32Array(labels.length);
  let maxL = 0; for (let i = 0; i < labels.length; i++) { const v = remap[labels[i]]; merged[i] = v; if (v > maxL) maxL = v; }
  if (maxL === 0) return merged;
  const counts = new Int32Array(maxL + 1); for (let i = 0; i < merged.length; i++) counts[merged[i]]++;
  const out = new Int32Array(maxL + 1); let nxt = 1;
  for (let l = 1; l <= maxL; l++) if (counts[l] >= P.min_size) out[l] = nxt++;
  const res = new Int32Array(labels.length); for (let i = 0; i < merged.length; i++) res[i] = out[merged[i]];
  return res;
}

// ---- preprocessing ----
export function percentileNormalize(rgb, H, W, p = 0.1, eps = 1e-3) {
  // rgb: [3,H,W]; per-channel (x - p0.1)/max(eps, p99.9 - p0.1)
  const out = new Float32Array(3 * H * W);
  for (let c = 0; c < 3; c++) {
    const off = c * H * W, s = Float32Array.from(rgb.subarray(off, off + H * W)).sort();
    const lo = s[Math.floor(p / 100 * (s.length - 1))], hi = s[Math.floor((100 - p) / 100 * (s.length - 1))];
    const denom = Math.max(eps, hi - lo);
    for (let i = 0; i < H * W; i++) out[off + i] = (rgb[off + i] - lo) / denom;
  }
  return out;
}
export function padTo8(x, H, W) {
  const div = 8, Hp = Math.ceil(H / div) * div, Wp = Math.ceil(W / div) * div;
  const data = new Float32Array(3 * Hp * Wp);
  for (let c = 0; c < 3; c++) {                              // copy + symmetric (edge-included) reflect pad
    const si = c * H * W, di = c * Hp * Wp;
    for (let y = 0; y < H; y++) for (let xx = 0; xx < W; xx++) data[di + y * Wp + xx] = x[si + y * W + xx];
    for (let y = H; y < Hp; y++) for (let xx = 0; xx < W; xx++) data[di + y * Wp + xx] = data[di + Math.max(0, 2 * H - 1 - y) * Wp + xx];
    for (let y = 0; y < Hp; y++) for (let xx = W; xx < Wp; xx++) data[di + y * Wp + xx] = data[di + y * Wp + Math.max(0, 2 * W - 1 - xx)];
  }
  return { data, Hp, Wp };
}
function now() { return (typeof performance !== "undefined" ? performance.now() : Date.now()); }
