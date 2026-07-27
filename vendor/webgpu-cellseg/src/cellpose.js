// Cellpose cyto3 (CPnet) inference on WebGPU — environment-agnostic core.
// Runs in Deno (headless, for tests) and the browser (identical code).
// The forward pass is hand-written WGSL; the flow dynamics run in JS (CPU).
//
// Public API:
//   const cp = await CellposeWebGPU.create();
//   cp.loadWeights(manifestObj, binArrayBuffer);
//   const {output} = await cp.forwardFromInput(inputF32, H, W);   // [3,H,W]
//   const labels = cp.computeMasks(dP, cellprob, H, W, opts);      // Int32Array
//   const {labels, timings} = await cp.segmentImage(gray, H, W, opts);
//
// `gray` is the channel to segment. cyto3 also accepts an optional nuclear
// channel — cellpose's channels=[cyto, nuc] — passed as `opts.chan2` (a second
// [H,W] array in the same raw units). Omit it for cellpose's grayscale mode
// (channels=[0,0]), which is a fully supported mode, not a degraded one.
//
// Checkpoint cp001: naive shaders (one thread per output element), fresh
// buffers per op, style readback mid-forward. Correctness first.

// cp003/cp004: small-footprint shared-memory tiled conv, per input channel — a 16×16
// workgroup loads an activated input tile plus a BLK×K×K weight slab into shared memory
// and accumulates BLK output channels from there. (cp005's larger CHUNK tiling was
// reverted; it cut occupancy.) Phase 1 kept that structure and fixed what was inside it.
import { requestDevice } from "./device.js";

// The conv kernel is generated per kernel size in src/conv-kernel.js. Phase 1 rebuilt it
// — 10.9x faster, bit-identical output, 3.5% → 37% of this machine's roof — and that file
// documents why each piece is shaped the way it is. src/profile/baseline-conv.js keeps
// the version it replaced, as the benchmark's zero point.
import { convWGSL, convDispatch } from "./conv-kernel.js";

const ADD_WGSL = /* wgsl */`
struct P { N:u32, nwgx:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       a: array<f32>;
@group(0) @binding(2) var<storage,read>       b: array<f32>;
@group(0) @binding(3) var<storage,read_write> o: array<f32>;
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let i = (wg.y * p.nwgx + wg.x) * 256u + li.x;
  if (i >= p.N) { return; }
  o[i] = a[i] + b[i];
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

// global average pool over spatial dims -> [C]
const GAP_WGSL = /* wgsl */`
struct P { H:u32, W:u32, C:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       inp: array<f32>;
@group(0) @binding(2) var<storage,read_write> outp: array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let c = gid.x;
  if (c >= p.C) { return; }
  let HW = p.H * p.W;
  var s = 0.0;
  for (var i = 0u; i < HW; i = i + 1u) { s = s + inp[c * HW + i]; }
  outp[c] = s / f32(HW);
}`;

// cp005: Euler integration of the flow dynamics on the GPU. One thread per seed
// pixel runs all niter steps (bilinear-interpolated flow, clamped), replacing
// the O(niter × npts) JS loop. Histogram/labeling stays on the CPU.
const STEPS_WGSL = /* wgsl */`
struct P { npts:u32, niter:u32, H:u32, W:u32, nwgx:u32, _pad:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       dy: array<f32>;   // [H*W] masked flow/5 (y)
@group(0) @binding(2) var<storage,read>       dx: array<f32>;   // [H*W] masked flow/5 (x)
@group(0) @binding(3) var<storage,read_write> pos: array<f32>;  // [2*npts]: py[0..npts), px[npts..)
@compute @workgroup_size(256)
fn main(@builtin(workgroup_id) wg: vec3<u32>, @builtin(local_invocation_id) li: vec3<u32>) {
  let i = (wg.y * p.nwgx + wg.x) * 256u + li.x;
  if (i >= p.npts) { return; }
  let Hm = f32(p.H - 1u); let Wm = f32(p.W - 1u);
  var py = pos[i]; var px = pos[p.npts + i];
  for (var t = 0u; t < p.niter; t = t + 1u) {
    let yf = floor(py); let xf = floor(px);
    let fy = py - yf; let fx = px - xf;
    var yi = i32(yf); var xi = i32(xf);
    yi = clamp(yi, 0, i32(Hm)); xi = clamp(xi, 0, i32(Wm));
    let yi1 = min(yi + 1, i32(Hm)); let xi1 = min(xi + 1, i32(Wm));
    let w00 = (1.0 - fy) * (1.0 - fx); let w01 = (1.0 - fy) * fx;
    let w10 = fy * (1.0 - fx); let w11 = fy * fx;
    let i00 = u32(yi) * p.W + u32(xi); let i01 = u32(yi) * p.W + u32(xi1);
    let i10 = u32(yi1) * p.W + u32(xi); let i11 = u32(yi1) * p.W + u32(xi1);
    let sdy = dy[i00]*w00 + dy[i01]*w01 + dy[i10]*w10 + dy[i11]*w11;
    let sdx = dx[i00]*w00 + dx[i01]*w01 + dx[i10]*w10 + dx[i11]*w11;
    py = clamp(py + sdy, 0.0, Hm);
    px = clamp(px + sdx, 0.0, Wm);
  }
  pos[i] = py; pos[p.npts + i] = px;
}`;

// Flow-consistency QC, on the GPU.
//
// cellpose's remove_bad_flow_masks reconstructs each mask's flow field by running a
// 9-point diffusion from a heat source at the mask's centre pixel, then compares the
// gradient of that field against the network's predicted flow. Phase 1 measurement
// (docs/PHASE1.md) found this single step at 95% of mask assembly and 72% of total wall
// clock once the conv kernel got 11x faster — by far the largest remaining line item.
//
// The CPU version runs one mask at a time over its own bounding box. This runs every
// mask at once over the whole image, which is equivalent given two details:
//
//   - **Neighbours are label-masked.** Per-mask, the diffusion field is sized to one
//     bounding box and positions outside the mask footprint stay zero forever. Globally,
//     a neighbouring pixel of a *different* mask holds that mask's heat, so it must be
//     read as zero. Without this two adjacent cells bleed into each other.
//   - **The source is folded into the read.** The CPU version does `T[med] += 1` at the
//     top of each iteration, so the increment accumulates into the field. Adding 1 to the
//     centre pixel's value as it is read is algebraically the same thing: the new centre
//     value already contains the increment, and the next iteration adds another.
//
// Each mask runs its own iteration count (2*(ly+lx)), so masks freeze individually once
// they reach it and the dispatch loop runs to the maximum. Freezing is per label and
// neighbours are same-label by construction, so no mask ever reads a half-frozen one.
const FLOWDIFF_WGSL = /* wgsl */`
struct P { H:u32, W:u32, t:i32, _p:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       labels: array<i32>;
@group(0) @binding(2) var<storage,read>       niter:  array<i32>;
@group(0) @binding(3) var<storage,read>       med:    array<i32>;
@group(0) @binding(4) var<storage,read>       Tin:    array<f32>;
@group(0) @binding(5) var<storage,read_write> Tout:   array<f32>;

fn samp(x: i32, y: i32, l: i32, m: i32) -> f32 {
  if (x < 0 || x >= i32(p.W) || y < 0 || y >= i32(p.H)) { return 0.0; }
  let n = y * i32(p.W) + x;
  if (labels[n] != l) { return 0.0; }   // a different mask's heat is not ours to read
  var v = Tin[n];
  if (n == m) { v = v + 1.0; }          // the heat source, folded into the read
  return v;
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.W || gid.y >= p.H) { return; }
  let i = i32(gid.y * p.W + gid.x);
  let l = labels[i];
  if (l <= 0) { Tout[i] = 0.0; return; }
  if (p.t >= niter[l]) { Tout[i] = Tin[i]; return; }
  let x = i32(gid.x); let y = i32(gid.y);
  let m = med[l];
  // Summation order matches the CPU implementation tap for tap.
  let s = samp(x, y, l, m)
        + samp(x, y - 1, l, m) + samp(x, y + 1, l, m)
        + samp(x - 1, y, l, m) + samp(x + 1, y, l, m)
        + samp(x - 1, y - 1, l, m) + samp(x + 1, y - 1, l, m)
        + samp(x - 1, y + 1, l, m) + samp(x + 1, y + 1, l, m);
  Tout[i] = s / 9.0;
}`;

// Central difference of the diffused field, label-masked for the same reason as above.
// Normalisation and the per-mask error stay on the CPU in f64: they are O(HW) and cheap,
// and keeping them in double means the threshold comparison that decides how many masks
// survive is not made on f32 sums.
const FLOWGRAD_WGSL = /* wgsl */`
struct P { H:u32, W:u32, t:i32, _p:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       labels: array<i32>;
@group(0) @binding(4) var<storage,read>       T:      array<f32>;
@group(0) @binding(5) var<storage,read_write> grad:   array<f32>;

fn g(x: i32, y: i32, l: i32) -> f32 {
  if (x < 0 || x >= i32(p.W) || y < 0 || y >= i32(p.H)) { return 0.0; }
  let n = y * i32(p.W) + x;
  if (labels[n] != l) { return 0.0; }
  return T[n];
}

@compute @workgroup_size(16,16,1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  if (gid.x >= p.W || gid.y >= p.H) { return; }
  let i = i32(gid.y * p.W + gid.x);
  let HW = i32(p.H * p.W);
  let l = labels[i];
  if (l <= 0) { grad[i] = 0.0; grad[HW + i] = 0.0; return; }
  let x = i32(gid.x); let y = i32(gid.y);
  grad[i]      = g(x, y + 1, l) - g(x, y - 1, l);
  grad[HW + i] = g(x + 1, y, l) - g(x - 1, y, l);
}`;

// cp006: L2-normalize the style vector on the GPU (one workgroup reduction),
// so the forward no longer stalls on a CPU readback mid-pass.
const NORMSTYLE_WGSL = /* wgsl */`
@group(0) @binding(0) var<storage,read>       raw: array<f32>;
@group(0) @binding(1) var<storage,read_write> out: array<f32>;
var<workgroup> red: array<f32, 256>;
@compute @workgroup_size(256)
fn main(@builtin(local_invocation_id) lid: vec3<u32>) {
  let i = lid.x; let v = raw[i]; red[i] = v * v;
  workgroupBarrier();
  for (var s = 128u; s > 0u; s = s >> 1u) {
    if (i < s) { red[i] = red[i] + red[i + s]; }
    workgroupBarrier();
  }
  out[i] = v * inverseSqrt(red[0]);
}`;

// cp006: style projection on the GPU. effShift[o] = scale[o]*(full_w[o]·style +
// full_b[o]) + shift[o]. Replaces the ~800K-mult/forward JS loop and keeps the
// whole forward on the GPU in a single command encoder.
const STYLEPROJ_WGSL = /* wgsl */`
struct P { Cout:u32, S:u32 };
@group(0) @binding(0) var<uniform> p: P;
@group(0) @binding(1) var<storage,read>       style: array<f32>;
@group(0) @binding(2) var<storage,read>       fw:    array<f32>;
@group(0) @binding(3) var<storage,read>       fb:    array<f32>;
@group(0) @binding(4) var<storage,read>       scale: array<f32>;
@group(0) @binding(5) var<storage,read>       shift: array<f32>;
@group(0) @binding(6) var<storage,read_write> eff:   array<f32>;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
  let o = gid.x;
  if (o >= p.Cout) { return; }
  var s = fb[o];
  let base = o * p.S;
  for (var k = 0u; k < p.S; k = k + 1u) { s = s + fw[base + k] * style[k]; }
  eff[o] = scale[o] * s + shift[o];
}`;

const NBASE = [2, 32, 64, 128, 256];
const NBASEUP = [32, 64, 128, 256, 256];

export class CellposeWebGPU {
  constructor(device) {
    this.device = device;
    const d = device;
    const mk = (code) => d.createComputePipeline({
      layout: "auto", compute: { module: d.createShaderModule({ code }), entryPoint: "main" }
    });
    // One conv pipeline per kernel size. K is a compile-time constant in the shader so
    // the taps unroll and the weight addressing folds — worth 1.4x on its own, and the
    // network only ever uses K∈{1,3}, so it costs two pipelines.
    this.pConvK = new Map([[1, mk(convWGSL(1))], [3, mk(convWGSL(3))]]);
    this.pAdd = mk(ADD_WGSL);
    this.pPool = mk(POOL_WGSL);
    this.pUp = mk(UP_WGSL);
    this.pGap = mk(GAP_WGSL);
    this.pSteps = mk(STEPS_WGSL);
    this.pNorm = mk(NORMSTYLE_WGSL);
    this.pProj = mk(STYLEPROJ_WGSL);
    // The flow-QC diffusion runs one dispatch per iteration inside a single encoder, and
    // each needs its own `t`. An explicit layout with a dynamic offset lets all of them
    // share one uniform buffer, which is why these two can't use `mk`'s auto layout.
    this._flowLayout = d.createBindGroupLayout({
      entries: [
        { binding: 0, visibility: GPUShaderStage.COMPUTE, buffer: { type: "uniform", hasDynamicOffset: true } },
        { binding: 1, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 2, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 3, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 4, visibility: GPUShaderStage.COMPUTE, buffer: { type: "read-only-storage" } },
        { binding: 5, visibility: GPUShaderStage.COMPUTE, buffer: { type: "storage" } },
      ],
    });
    const flowPl = d.createPipelineLayout({ bindGroupLayouts: [this._flowLayout] });
    const mkFlow = (code) => d.createComputePipeline({
      layout: flowPl, compute: { module: d.createShaderModule({ code }), entryPoint: "main" },
    });
    this.pFlowDiff = mkFlow(FLOWDIFF_WGSL);
    this.pFlowGrad = mkFlow(FLOWGRAD_WGSL);
    this.buf = {};        // static weight buffers by name
    this._pool = new Map();  // free-list keyed by usage:size
    this._inUse = [];        // buffers acquired during the current forward
  }

  static async create() {
    return new CellposeWebGPU(await requestDevice());
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
    this.dummy = d.createBuffer({ size: 16, usage: U });
  }

  // Convenience loader: fetch manifest.json + weights.bin from a base URL and
  // return a ready-to-use instance. The default resolves to this repo's
  // weights/cellpose-cyto3/ folder relative to THIS module (works no matter where
  // the importing page lives). Pass a base URL — e.g. a HuggingFace or jsDelivr
  // URL — to load the weights from a CDN instead:
  //   const cp = await CellposeWebGPU.load("https://huggingface.co/<you>/cellpose-webgpu/resolve/main/cellpose-cyto3/");
  // Pass `device` to build on an already-acquired GPUDevice instead of requesting a
  // second one — how a host shares a single device across several models.
  static async load(baseURL = new URL("../weights/cellpose-cyto3/", import.meta.url).href,
                    { manifest = "manifest.json", weights = "weights.bin", device = null } = {}) {
    const base = baseURL.endsWith("/") ? baseURL : baseURL + "/";
    const [mf, bin] = await Promise.all([
      fetch(base + manifest).then(r => r.json()),
      fetch(base + weights).then(r => r.arrayBuffer()),
    ]);
    const inst = device ? new this(device) : await this.create();
    inst.loadWeights(mf, bin);
    return inst;
  }

  arr(name) {
    const t = this.tensors[name];
    return this.blob.subarray(t.offset, t.offset + t.length);
  }

  // Every command encoder this engine builds goes through here. The indirection exists
  // so src/profile/timing.js can substitute a wrapper that adds timestamp queries to
  // each compute pass without any op method learning about profiling — see the note in
  // that file about keeping the engines near-frozen. Unprofiled, this is exactly
  // `device.createCommandEncoder()`.
  _mkEncoder(label) { return this.device.createCommandEncoder({ label }); }

  // Stage accounting for the Phase 0 Amdahl split. `_stages` is null unless a profiling
  // run sets it, so the unprofiled path pays for one null check per stage boundary and
  // never calls the clock. The buckets deliberately separate CPU work from GPU waits:
  // the coarse `timings` this returns alongside lumps the JS tile-blending in with the
  // forward pass and the JS mask assembly in with the dynamics kernel, which is exactly
  // the conflation Phase 0 exists to undo.
  _stage(name, t0) {
    if (!this._stages) return;
    this._stages[name] = (this._stages[name] || 0) + (now() - t0);
  }
  _mark() { return this._stages ? now() : 0; }

  // ---- buffer helpers (cp004: pooled, reused across ops and forwards) ----
  // Buffers are acquired from a free-list keyed by (usage,size) and returned to
  // the pool at the end of each forward instead of being destroyed — this
  // eliminates the per-op allocate/zero/free churn (hundreds of MB per call).
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
      GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST,
      nfloats * 4);
  }
  uniform(ints) {
    const b = this._acquire(GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST, ints.length * 4);
    this.device.queue.writeBuffer(b, 0, new Uint32Array(ints));
    return b;
  }
  readback(sizeBytes) {
    return this._acquire(GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ, sizeBytes);
  }
  freeScratch() { this.releaseAll(); }

  // ---- ops (record into encoder) ----
  // `name` is carried only to label the compute pass. The label is what src/profile/
  // parses back into a cost descriptor (FLOPs, ideal traffic, quantization waste), and
  // it shows up in devtools' GPU capture besides — so the shape has to stay in sync
  // with parseLabel() in src/profile/cost.js.
  conv(enc, { inBuf, outBuf, H, W, Cin, Cout, K, relu, addBuf, residBuf, wBuf, bBuf, scaleBuf, shiftBuf, name }) {
    const uni = this.uniform([H, W, Cin, Cout, K, (K / 2) | 0, relu ? 1 : 0, addBuf ? 1 : 0,
      residBuf ? 1 : 0, 0, 0, 0]);
    const pipe = this.pConvK.get(K);
    if (!pipe) throw new Error(`no conv pipeline for K=${K} (built: ${[...this.pConvK.keys()]})`);
    const bg = this.device.createBindGroup({
      layout: pipe.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: wBuf } },
        { binding: 3, resource: { buffer: bBuf } },
        { binding: 4, resource: { buffer: scaleBuf } },
        { binding: 5, resource: { buffer: shiftBuf } },
        { binding: 6, resource: { buffer: addBuf || this.dummy } },
        { binding: 8, resource: { buffer: residBuf || this.dummy } },
        { binding: 7, resource: { buffer: outBuf } },
      ]
    });
    const flags = (relu ? "|relu" : "") + (addBuf ? "|add" : "") + (residBuf ? "|resid" : "");
    const pass = enc.beginComputePass({
      label: `conv|${name || "?"}|${Cin}->${Cout}|${H}x${W}|k${K}${flags}` });
    pass.setPipeline(pipe); pass.setBindGroup(0, bg);
    // Each thread now covers an RBY x RBX block of output pixels, so the grid shrinks by
    // that factor — convDispatch owns the arithmetic so it cannot drift from the shader.
    const [gx, gy, gz] = convDispatch(H, W, Cout);
    pass.dispatchWorkgroups(gx, gy, gz);
    pass.end();
  }
  add(enc, aBuf, bBuf, outBuf, N) {
    const nwg = Math.ceil(N / 256);
    const nwgx = Math.min(nwg, 65535), nwgy = Math.ceil(nwg / nwgx);
    const uni = this.uniform([N, nwgx]);
    const bg = this.device.createBindGroup({
      layout: this.pAdd.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: aBuf } },
        { binding: 2, resource: { buffer: bBuf } },
        { binding: 3, resource: { buffer: outBuf } },
      ]
    });
    const pass = enc.beginComputePass({ label: `add|N${N}` });
    pass.setPipeline(this.pAdd); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(nwgx, nwgy); pass.end();
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
    const pass = enc.beginComputePass({ label: `pool|${Hi}x${Wi}->${Ho}x${Wo}|C${C}` });
    pass.setPipeline(this.pPool); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Wo / 16), Math.ceil(Ho / 16), C); pass.end();
    return [Ho, Wo];
  }
  up(enc, inBuf, outBuf, Hi, Wi, C) {
    const Ho = Hi * 2, Wo = Wi * 2;
    const uni = this.uniform([Ho, Wo, Hi, Wi, C]);
    const bg = this.device.createBindGroup({
      layout: this.pUp.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ]
    });
    const pass = enc.beginComputePass({ label: `up|${Hi}x${Wi}->${Ho}x${Wo}|C${C}` });
    pass.setPipeline(this.pUp); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Wo / 16), Math.ceil(Ho / 16), C); pass.end();
    return [Ho, Wo];
  }
  gap(enc, inBuf, outBuf, H, W, C) {
    const uni = this.uniform([H, W, C]);
    const bg = this.device.createBindGroup({
      layout: this.pGap.getBindGroupLayout(0), entries: [
        { binding: 0, resource: { buffer: uni } },
        { binding: 1, resource: { buffer: inBuf } },
        { binding: 2, resource: { buffer: outBuf } },
      ]
    });
    const pass = enc.beginComputePass({ label: `gap|${H}x${W}|C${C}` });
    pass.setPipeline(this.pGap); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(C / 64)); pass.end();
  }

  // batchconv helper: picks static scale/shift buffers by name.
  // addBuf sums into the input (pre-BN); residBuf sums into the output (residual).
  bc(enc, name, inBuf, outBuf, H, W, Cin, Cout, K, relu, addBuf, shiftBufOverride, residBuf) {
    this.conv(enc, {
      inBuf, outBuf, H, W, Cin, Cout, K, relu, addBuf, residBuf, name,
      wBuf: this.buf[name + ".w"], bBuf: this.buf[name + ".b"],
      scaleBuf: this.buf[name + ".scale"],
      shiftBuf: shiftBufOverride || this.buf[name + ".shift"],
    });
  }

  resdown(enc, n, x, Cin, Cout, H, W) {
    const d = `down.${n}`, HW = Cout * H * W;
    const a0 = this.mkStorage(HW), pj = this.mkStorage(HW),
      x1 = this.mkStorage(HW), a2 = this.mkStorage(HW), x2 = this.mkStorage(HW);
    this.bc(enc, `${d}.conv0`, x, a0, H, W, Cin, Cout, 3, true);
    this.bc(enc, `${d}.proj`, x, pj, H, W, Cin, Cout, 1, false);
    // x1 = proj(x) + conv1(a0)   — residual fused into conv1's output
    this.bc(enc, `${d}.conv1`, a0, x1, H, W, Cout, Cout, 3, true, null, null, pj);
    this.bc(enc, `${d}.conv2`, x1, a2, H, W, Cout, Cout, 3, true);
    // x2 = x1 + conv3(a2)        — residual fused into conv3's output
    this.bc(enc, `${d}.conv3`, a2, x2, H, W, Cout, Cout, 3, true, null, null, x1);
    return x2;
  }

  // GPU L2-normalize of the style vector (raw [256] -> normalized [256]).
  normStyle(enc, rawBuf, outBuf) {
    const bg = this.device.createBindGroup({ layout: this.pNorm.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: rawBuf } }, { binding: 1, resource: { buffer: outBuf } }] });
    const pass = enc.beginComputePass({ label: "normstyle|C256" });
    pass.setPipeline(this.pNorm); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(1); pass.end();
  }

  // GPU style projection: effShift = scale*(full_w·style + full_b) + shift, on device.
  effShiftGPU(enc, name, styleBuf, Cout) {
    const eff = this.mkStorage(Cout);
    const uni = this.uniform([Cout, 256]);
    const bg = this.device.createBindGroup({ layout: this.pProj.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: styleBuf } },
      { binding: 2, resource: { buffer: this.buf[name + ".full.w"] } },
      { binding: 3, resource: { buffer: this.buf[name + ".full.b"] } },
      { binding: 4, resource: { buffer: this.buf[name + ".scale"] } },
      { binding: 5, resource: { buffer: this.buf[name + ".shift"] } },
      { binding: 6, resource: { buffer: eff } }] });
    const pass = enc.beginComputePass({ label: `styleproj|${name}|C${Cout}` });
    pass.setPipeline(this.pProj); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(Math.ceil(Cout / 64)); pass.end();
    return eff;
  }

  resup(enc, n, x, y, styleBuf, Cin, Cout, H, W) {
    const d = `up.${n}`, HW = Cout * H * W;
    const es1 = this.effShiftGPU(enc, `${d}.conv1`, styleBuf, Cout);
    const es2 = this.effShiftGPU(enc, `${d}.conv2`, styleBuf, Cout);
    const es3 = this.effShiftGPU(enc, `${d}.conv3`, styleBuf, Cout);
    const a0 = this.mkStorage(HW), pj = this.mkStorage(HW),
      x1 = this.mkStorage(HW), a2 = this.mkStorage(HW), x2 = this.mkStorage(HW);
    this.bc(enc, `${d}.conv0`, x, a0, H, W, Cin, Cout, 3, true);
    this.bc(enc, `${d}.proj`, x, pj, H, W, Cin, Cout, 1, false);
    // x1 = proj(x) + conv1(a0 + y);  skip y added pre-BN, proj fused into output
    this.bc(enc, `${d}.conv1`, a0, x1, H, W, Cout, Cout, 3, true, y, es1, pj);
    this.bc(enc, `${d}.conv2`, x1, a2, H, W, Cout, Cout, 3, true, null, es2);
    // x2 = x1 + conv3(a2);  x1 residual fused into output
    this.bc(enc, `${d}.conv3`, a2, x2, H, W, Cout, Cout, 3, true, null, es3, x1);
    return x2;
  }

  // Run CPnet forward on a preprocessed input [2,H,W]. Returns Float32Array [3,H,W].
  async forwardFromInput(inputF32, H, W) {
    const d = this.device;
    const inBuf = this.mkStorage(2 * H * W);
    d.queue.writeBuffer(inBuf, 0, inputF32);

    // cp006: whole forward in a SINGLE command encoder — encoder, GPU style
    // (normalize + projections), decoder, output head. No mid-forward CPU stall.
    const enc = this._mkEncoder("forward");
    const xd = [];
    let y = inBuf, hy = H, wy = W;
    for (let n = 0; n < 4; n++) {
      if (n > 0) {
        const C = NBASE[n]; // channels of xd[n-1]
        const pooled = this.mkStorage(C * (hy >> 1) * (wy >> 1));
        [hy, wy] = this.pool(enc, xd[n - 1], pooled, hy, wy, C);
        y = pooled;
      }
      xd[n] = this.resdown(enc, n, y, NBASE[n], NBASE[n + 1], hy, wy);
    }
    // style vector on the GPU (global-avg-pool -> L2 normalize)
    const styleRaw = this.mkStorage(256);
    this.gap(enc, xd[3], styleRaw, hy, wy, 256);
    const styleBuf = this.mkStorage(256);
    this.normStyle(enc, styleRaw, styleBuf);

    // decoder — resup blocks compute their effShift on the GPU from styleBuf
    // xd resolutions: xd[3] at (hy,wy); xd[2] at (2h,2w); xd[1] at (4h,4w); xd[0] at (8h,8w)
    const res = [[H, W], [H >> 1, W >> 1], [H >> 2, W >> 2], [H >> 3, W >> 3]];
    let x = this.resup(enc, 3, xd[3], xd[3], styleBuf, 256, 256, res[3][0], res[3][1]);
    const Cup = { 2: [256, 128], 1: [128, 64], 0: [64, 32] };
    let ch = 256, ch_h = res[3][0], ch_w = res[3][1];
    for (const n of [2, 1, 0]) {
      const upBuf = this.mkStorage(ch * (ch_h * 2) * (ch_w * 2));
      const [nh, nw] = this.up(enc, x, upBuf, ch_h, ch_w, ch);
      const [cin, cout] = Cup[n];
      x = this.resup(enc, n, upBuf, xd[n], styleBuf, cin, cout, nh, nw);
      ch = cout; ch_h = nh; ch_w = nw;
    }
    // output head: batchconv 32->3, 1x1, relu
    const outBuf = this.mkStorage(3 * H * W);
    this.bc(enc, "output", x, outBuf, H, W, 32, 3, 1, true);
    const outRB = this.readback(3 * H * W * 4);
    enc.copyBufferToBuffer(outBuf, 0, outRB, 0, 3 * H * W * 4);
    d.queue.submit([enc.finish()]);
    await outRB.mapAsync(GPUMapMode.READ);
    const output = new Float32Array(outRB.getMappedRange().slice(0));
    outRB.unmap();
    this.releaseAll();
    return { output };
  }

  // Tiled forward, exact mirror of cellpose core.run_net. Zero-pads the
  // working-resolution channels (get_pad_yx), cuts overlapping 224×224 tiles,
  // runs forwardFromInput *per tile* — so each tile's decoder is conditioned by
  // a style vector computed from its own local view, which is the whole point —
  // then taper-blends the tile outputs (average_tiles) and crops the padding off.
  //   ch0/ch1: Float32Array [Lyr*Lxr] normalized+resized channels (ch1 may be null).
  //   returns Float32Array [3*Lyr*Lxr] = (dY,dX,cellprob) at the working resolution.
  async runNet(ch0, ch1, Lyr, Lxr) {
    const mPad = this._mark();
    const [ypad1, ypad2, xpad1, xpad2] = getPadYX(Lyr, Lxr, 16, 1);
    const Ly = Lyr + ypad1 + ypad2, Lx = Lxr + xpad1 + xpad2;
    // zero-padded 2-channel image [2,Ly,Lx] (cellpose pads with mode="constant")
    const img = new Float32Array(2 * Ly * Lx);
    for (let c = 0; c < 2; c++) {
      const src = c === 0 ? ch0 : ch1;
      if (!src) continue;
      const off = c * Ly * Lx;
      for (let y = 0; y < Lyr; y++) {
        const s = y * Lxr, d = off + (ypad1 + y) * Lx + xpad1;
        for (let x = 0; x < Lxr; x++) img[d + x] = src[s + x];
      }
    }
    this._stage("pad", mPad);
    const { ny, nx, bsizeY, bsizeX, ystart, xstart } = tileGrid(Ly, Lx, 224, 0.1);
    const TP = bsizeY * bsizeX;
    if (this._stages) this._stages.tiles = ny * nx;
    // extract one [2,bsizeY,bsizeX] tile at padded-image corner (y0,x0)
    const extract = (y0, x0) => {
      const m = this._mark();
      const tile = new Float32Array(2 * TP);
      for (let c = 0; c < 2; c++) {
        const so = c * Ly * Lx, to = c * TP;
        for (let ty = 0; ty < bsizeY; ty++) {
          const s = so + (y0 + ty) * Lx + x0, t = to + ty * bsizeX;
          for (let tx = 0; tx < bsizeX; tx++) tile[t + tx] = img[s + tx];
        }
      }
      this._stage("tile_extract", m);
      return tile;
    };
    // crop a padded [3,Ly,Lx] field back to [3,Lyr,Lxr]
    const cropPad = (yf) => {
      const m = this._mark();
      const out = new Float32Array(3 * Lyr * Lxr);
      for (let c = 0; c < 3; c++) {
        const so = c * Ly * Lx, to = c * Lyr * Lxr;
        for (let y = 0; y < Lyr; y++) {
          const s = so + (ypad1 + y) * Lx + xpad1, t = to + y * Lxr;
          for (let x = 0; x < Lxr; x++) out[t + x] = yf[s + x];
        }
      }
      this._stage("crop", m);
      return out;
    };
    // single tile spans the whole padded image; average_tiles' taper mask cancels
    // (yf = y[0]*mask/mask), so this is a plain forward — today's fast path.
    if (ny === 1 && nx === 1) {
      const tile = extract(ystart[0], xstart[0]);
      const mFwd = this._mark();
      const { output } = await this.forwardFromInput(tile, bsizeY, bsizeX);
      this._stage("forward_wait", mFwd);
      return cropPad(output);
    }
    // multi-tile: weighted accumulate with the taper mask, then divide by weight.
    const mask = taperMask(bsizeY, bsizeX);
    const acc = new Float64Array(3 * Ly * Lx);   // accumulate in f64 (numpy does)
    const Navg = new Float64Array(Ly * Lx);
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const y0 = ystart[j], x0 = xstart[i];
        const tile = extract(y0, x0);
        // Each tile is a full submit + mapAsync round-trip: the pipeline drains between
        // tiles even though tiles are entirely independent of one another. Measuring
        // this wait against the GPU-busy time inside it is how Phase 0 prices the
        // per-tile serialization.
        const mFwd = this._mark();
        const { output } = await this.forwardFromInput(tile, bsizeY, bsizeX);
        this._stage("forward_wait", mFwd);
        const mBlend = this._mark();
        for (let ty = 0; ty < bsizeY; ty++) {
          for (let tx = 0; tx < bsizeX; tx++) {
            const m = mask[ty * bsizeX + tx];
            const p = (y0 + ty) * Lx + (x0 + tx);
            Navg[p] += m;
            const ti = ty * bsizeX + tx;
            acc[p] += output[ti] * m;
            acc[Ly * Lx + p] += output[TP + ti] * m;
            acc[2 * Ly * Lx + p] += output[2 * TP + ti] * m;
          }
        }
        this._stage("blend", mBlend);
      }
    }
    const mNorm = this._mark();
    const yf = new Float32Array(3 * Ly * Lx);
    const N = Ly * Lx;
    for (let p = 0; p < N; p++) {
      const inv = 1 / Navg[p];
      yf[p] = acc[p] * inv;
      yf[N + p] = acc[N + p] * inv;
      yf[2 * N + p] = acc[2 * N + p] * inv;
    }
    this._stage("blend", mNorm);
    return cropPad(yf);
  }

  // ---- flow dynamics (JS port of cellpose compute_masks, CPU/interp path) ----
  // Common setup: seed pixels (cellprob>thr) and the masked flow field dP/5.
  _dynamicsSetup(dP, cellprob, H, W, cellprob_threshold) {
    const ys = [], xs = [];
    for (let yy = 0; yy < H; yy++) for (let xx = 0; xx < W; xx++)
      if (cellprob[yy * W + xx] > cellprob_threshold) { ys.push(yy); xs.push(xx); }
    const HW = H * W;
    const dy = new Float32Array(HW), dx = new Float32Array(HW);
    for (let i = 0; i < HW; i++) {
      if (cellprob[i] > cellprob_threshold) { dy[i] = dP[i] / 5; dx[i] = dP[HW + i] / 5; }
    }
    return { ys, xs, dy, dx };
  }

  // GPU Euler integration variant (cp005). Same result as computeMasks.
  async computeMasksGPU(dP, cellprob, H, W, opts = {}) {
    const { cellprob_threshold = 0.0, niter = 200, min_size = 15, rpad = 20, flow_threshold = 0.4 } = opts;
    const mSetup = this._mark();
    const { ys, xs, dy, dx } = this._dynamicsSetup(dP, cellprob, H, W, cellprob_threshold);
    this._stage("dyn_setup", mSetup);
    const npts = ys.length;
    if (npts === 0) return new Int32Array(H * W);
    const d = this.device, HW = H * W;
    const U = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const dyBuf = d.createBuffer({ size: HW * 4, usage: U });
    const dxBuf = d.createBuffer({ size: HW * 4, usage: U });
    const posBuf = d.createBuffer({ size: 2 * npts * 4, usage: U });
    d.queue.writeBuffer(dyBuf, 0, dy); d.queue.writeBuffer(dxBuf, 0, dx);
    const pos0 = new Float32Array(2 * npts);
    for (let i = 0; i < npts; i++) { pos0[i] = ys[i]; pos0[npts + i] = xs[i]; }
    d.queue.writeBuffer(posBuf, 0, pos0);
    const nwg = Math.ceil(npts / 256), nwgx = Math.min(nwg, 65535), nwgy = Math.ceil(nwg / nwgx);
    const uni = d.createBuffer({ size: 24, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
    d.queue.writeBuffer(uni, 0, new Uint32Array([npts, niter, H, W, nwgx, 0]));
    const bg = d.createBindGroup({ layout: this.pSteps.getBindGroupLayout(0), entries: [
      { binding: 0, resource: { buffer: uni } }, { binding: 1, resource: { buffer: dyBuf } },
      { binding: 2, resource: { buffer: dxBuf } }, { binding: 3, resource: { buffer: posBuf } }] });
    const mDyn = this._mark();
    const enc = this._mkEncoder("dynamics");
    const pass = enc.beginComputePass({ label: `dynamics|npts${npts}|niter${niter}` });
    pass.setPipeline(this.pSteps); pass.setBindGroup(0, bg);
    pass.dispatchWorkgroups(nwgx, nwgy); pass.end();
    const rb = d.createBuffer({ size: 2 * npts * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(posBuf, 0, rb, 0, 2 * npts * 4);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const posF = new Float32Array(rb.getMappedRange().slice(0)); rb.unmap();
    dyBuf.destroy(); dxBuf.destroy(); posBuf.destroy(); uni.destroy(); rb.destroy();
    this._stage("dyn_wait", mDyn);
    const py = posF.subarray(0, npts), px = posF.subarray(npts, 2 * npts);
    const mMasks = this._mark();
    const raw = this._getMasks(py, px, ys, xs, dy, dx, H, W, rpad, min_size, flow_threshold);
    const labels = await this._filterMasksGPU(raw, dy, dx, H, W, min_size, flow_threshold);
    this._stage("getmasks", mMasks);
    return labels;
  }

  computeMasks(dP, cellprob, H, W, opts = {}) {
    const { cellprob_threshold = 0.0, niter = 200, min_size = 15, rpad = 20, flow_threshold = 0.4 } = opts;
    const { ys, xs, dy, dx } = this._dynamicsSetup(dP, cellprob, H, W, cellprob_threshold);
    const npts = ys.length;
    if (npts === 0) return new Int32Array(H * W);
    const HW = H * W;
    // Euler integration with bilinear interp (map_coordinates), clamp [0,H-1]/[0,W-1]
    const py = new Float32Array(npts), px = new Float32Array(npts);
    for (let i = 0; i < npts; i++) { py[i] = ys[i]; px[i] = xs[i]; }
    const Hm = H - 1, Wm = W - 1;
    for (let t = 0; t < niter; t++) {
      for (let i = 0; i < npts; i++) {
        const yc = py[i], xc = px[i];
        let yf = yc | 0, xf = xc | 0;
        if (yf < 0) yf = 0; else if (yf > Hm) yf = Hm;
        if (xf < 0) xf = 0; else if (xf > Wm) xf = Wm;
        const yf1 = yf < Hm ? yf + 1 : Hm, xf1 = xf < Wm ? xf + 1 : Wm;
        const fy = yc - (yc | 0), fx = xc - (xc | 0);
        const w00 = (1 - fy) * (1 - fx), w01 = (1 - fy) * fx, w10 = fy * (1 - fx), w11 = fy * fx;
        const i00 = yf * W + xf, i01 = yf * W + xf1, i10 = yf1 * W + xf, i11 = yf1 * W + xf1;
        const sdy = dy[i00] * w00 + dy[i01] * w01 + dy[i10] * w10 + dy[i11] * w11;
        const sdx = dx[i00] * w00 + dx[i01] * w01 + dx[i10] * w10 + dx[i11] * w11;
        let ny = yc + sdy, nx = xc + sdx;
        if (ny < 0) ny = 0; else if (ny > Hm) ny = Hm;
        if (nx < 0) nx = 0; else if (nx > Wm) nx = Wm;
        py[i] = ny; px[i] = nx;
      }
    }
    const raw = this._getMasks(py, px, ys, xs, dy, dx, H, W, rpad, min_size, flow_threshold);
    return this._filterMasks(raw, dy, dx, H, W, min_size, flow_threshold);
  }

  _getMasks(py, px, ys, xs, dy, dx, H, W, rpad, min_size, flow_threshold) {
    const mHist = this._mark();
    const npts = py.length;
    const Hh = H + 2 * rpad, Ww = W + 2 * rpad;
    // integer final positions (trunc) + rpad, clamp to [0, shape0+rpad-1]
    const pty = new Int32Array(npts), ptx = new Int32Array(npts);
    for (let i = 0; i < npts; i++) {
      let a = (py[i] | 0) + rpad, b = (px[i] | 0) + rpad;
      if (a < 0) a = 0; else if (a > H + rpad - 1) a = H + rpad - 1;
      if (b < 0) b = 0; else if (b > W + rpad - 1) b = W + rpad - 1;
      pty[i] = a; ptx[i] = b;
    }
    // histogram
    const h1 = new Int32Array(Hh * Ww);
    for (let i = 0; i < npts; i++) h1[pty[i] * Ww + ptx[i]]++;
    this._stage("gm_hist", mHist);
    // local maxima in 5x5 with count>10
    const mSeeds = this._mark();
    const seeds = [];
    for (let y = 0; y < Hh; y++) for (let x = 0; x < Ww; x++) {
      const c = h1[y * Ww + x];
      if (c <= 10) continue;
      let isMax = true;
      for (let dyk = -2; dyk <= 2 && isMax; dyk++) {
        const yy = y + dyk; if (yy < 0 || yy >= Hh) continue;
        for (let dxk = -2; dxk <= 2; dxk++) {
          const xx = x + dxk; if (xx < 0 || xx >= Ww) continue;
          if (h1[yy * Ww + xx] > c) { isMax = false; break; }
        }
      }
      if (isMax) seeds.push([y, x, c]);
    }
    if (seeds.length === 0) return new Int32Array(H * W);
    seeds.sort((a, b) => a[2] - b[2]); // ascending count; later (bigger) wins overlaps
    this._stage("gm_seeds", mSeeds);

    // grow each seed within 11x11 window: 5 iters dilate(3x3) & (h_slc>2)
    const mGrow = this._mark();
    const M1 = new Int32Array(Hh * Ww);
    for (let k = 0; k < seeds.length; k++) {
      const [sy, sx] = seeds[k];
      let cur = new Uint8Array(121); cur[5 * 11 + 5] = 1;
      const hslc = new Int32Array(121);
      for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) {
        const yy = sy - 5 + a, xx = sx - 5 + b;
        hslc[a * 11 + b] = (yy >= 0 && yy < Hh && xx >= 0 && xx < Ww) ? h1[yy * Ww + xx] : 0;
      }
      for (let it = 0; it < 5; it++) {
        const nxt = new Uint8Array(121);
        for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) {
          let v = 0;
          for (let da = -1; da <= 1 && !v; da++) for (let db = -1; db <= 1; db++) {
            const aa = a + da, bb = b + db;
            if (aa < 0 || aa >= 11 || bb < 0 || bb >= 11) continue;
            if (cur[aa * 11 + bb]) { v = 1; break; }
          }
          if (v && hslc[a * 11 + b] > 2) nxt[a * 11 + b] = 1;
        }
        cur = nxt;
      }
      const label = k + 1;
      for (let a = 0; a < 11; a++) for (let b = 0; b < 11; b++) {
        if (cur[a * 11 + b]) {
          const yy = sy - 5 + a, xx = sx - 5 + b;
          if (yy >= 0 && yy < Hh && xx >= 0 && xx < Ww) M1[yy * Ww + xx] = label;
        }
      }
    }
    this._stage("gm_grow", mGrow);
    // assign each pixel the label of its final bucket
    const labels = new Int32Array(H * W);
    for (let i = 0; i < npts; i++) {
      labels[ys[i] * W + xs[i]] = M1[pty[i] * Ww + ptx[i]];
    }
    // The caller filters: the GPU flow-QC path is async and the plain computeMasks()
    // entry point must stay synchronous, so _getMasks stops at the raw label map.
    return labels;
  }

  // Port of cellpose masks_to_flows_cpu: for each mask, run a 9-point diffusion
  // (heat source at the mask's center pixel) over the mask's own footprint, then take
  // the gradient of the diffused field as that mask's "reconstructed" flow. Returns the
  // per-mask mean squared error against the network's predicted flow (dy,dx = dP/5),
  // i.e. a JS port of cellpose.metrics.flow_error. Only computed for labels with keep[l].
  _maskFlowErrors(labels, dy, dx, H, W, maxLabel, keep) {
    const minY = new Int32Array(maxLabel + 1).fill(H), maxY = new Int32Array(maxLabel + 1).fill(-1);
    const minX = new Int32Array(maxLabel + 1).fill(W), maxX = new Int32Array(maxLabel + 1).fill(-1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const l = labels[y * W + x];
      if (l === 0) continue;
      if (y < minY[l]) minY[l] = y; if (y > maxY[l]) maxY[l] = y;
      if (x < minX[l]) minX[l] = x; if (x > maxX[l]) maxX[l] = x;
    }
    const dyMask = new Float64Array(H * W), dxMask = new Float64Array(H * W);
    for (let l = 1; l <= maxLabel; l++) {
      if (!keep[l] || maxY[l] < 0) continue;
      const ly = (maxY[l] - minY[l] + 1) + 2, lx = (maxX[l] - minX[l] + 1) + 2;
      const Y = [], X = [];
      for (let gy = minY[l]; gy <= maxY[l]; gy++) for (let gx = minX[l]; gx <= maxX[l]; gx++) {
        if (labels[gy * W + gx] === l) { Y.push(gy - minY[l] + 1); X.push(gx - minX[l] + 1); }
      }
      const n = Y.length;
      let ysum = 0, xsum = 0;
      for (let k = 0; k < n; k++) { ysum += Y[k]; xsum += X[k]; }
      const ymedf = ysum / n, xmedf = xsum / n;
      let imin = 0, best = Infinity;
      for (let k = 0; k < n; k++) {
        const dd = (X[k] - xmedf) ** 2 + (Y[k] - ymedf) ** 2;
        if (dd < best) { best = dd; imin = k; }
      }
      const ymed = Y[imin], xmed = X[imin];
      const niter = 2 * (ly + lx);
      const T = new Float64Array(ly * lx);
      const nxt = new Float64Array(n);
      // Flat index per mask pixel, computed once instead of rebuilt from (Y,X) on every
      // one of the ~2*(ly+lx) iterations. The nine taps then come off one base by
      // addition rather than three multiplies plus two typed-array loads per pixel.
      // The summation order is unchanged, so this is bit-identical — which matters,
      // because the result feeds a threshold comparison that decides how many masks
      // survive (173 of 190 on the reference image).
      const idx = new Int32Array(n);
      for (let k = 0; k < n; k++) idx[k] = Y[k] * lx + X[k];
      const medIdx = ymed * lx + xmed;
      for (let t = 0; t < niter; t++) {
        T[medIdx] += 1;
        for (let k = 0; k < n; k++) {
          const c = idx[k], u = c - lx, v = c + lx;
          nxt[k] = (T[c] + T[u] + T[v] +
                    T[c - 1] + T[c + 1] +
                    T[u - 1] + T[u + 1] +
                    T[v - 1] + T[v + 1]) / 9;
        }
        for (let k = 0; k < n; k++) T[idx[k]] = nxt[k];
      }
      for (let k = 0; k < n; k++) {
        const yl = Y[k], xl = X[k];
        const gdy = T[(yl + 1) * lx + xl] - T[(yl - 1) * lx + xl];
        const gdx = T[yl * lx + xl + 1] - T[yl * lx + xl - 1];
        const gy = minY[l] + (yl - 1), gx = minX[l] + (xl - 1);
        dyMask[gy * W + gx] = gdy; dxMask[gy * W + gx] = gdx;
      }
    }
    // per-pixel unit-normalize (matches cellpose's global mu /= |mu|)
    for (let i = 0; i < H * W; i++) {
      const mag = Math.sqrt(dyMask[i] * dyMask[i] + dxMask[i] * dxMask[i]);
      dyMask[i] /= (mag + 1e-60); dxMask[i] /= (mag + 1e-60);
    }
    // per-mask mean squared error vs network flow (dy,dx already = dP/5)
    const errSum = new Float64Array(maxLabel + 1), errN = new Int32Array(maxLabel + 1);
    for (let i = 0; i < H * W; i++) {
      const l = labels[i];
      if (l === 0 || !keep[l]) continue;
      const ed = dyMask[i] - dy[i], ex = dxMask[i] - dx[i];
      errSum[l] += ed * ed + ex * ex; errN[l]++;
    }
    const merr = new Float64Array(maxLabel + 1);
    for (let l = 1; l <= maxLabel; l++) if (errN[l] > 0) merr[l] = errSum[l] / errN[l];
    return merr;
  }

  /**
   * GPU flow-consistency QC. Same contract as _maskFlowErrors: returns per-label mean
   * squared error between the reconstructed and predicted flow.
   *
   * The diffusion runs on the GPU (see FLOWDIFF_WGSL for why the global formulation is
   * equivalent to the per-mask one); the normalisation and the per-mask error stay on
   * the CPU in f64, so the comparison against flow_threshold — which decides how many
   * masks survive — is not made on f32 sums.
   */
  async _maskFlowErrorsGPU(labels, dy, dx, H, W, maxLabel, keep) {
    const d = this.device, HW = H * W;
    // Bounding boxes, centroids and the centre pixel, in three linear scans instead of
    // one pass per mask. Local and global coordinates differ by a per-mask constant, so
    // the centroid argmin is the same either way — and raster order matches the CPU's
    // iteration order, which matters only for tie-breaking.
    const minY = new Int32Array(maxLabel + 1).fill(H), maxY = new Int32Array(maxLabel + 1).fill(-1);
    const minX = new Int32Array(maxLabel + 1).fill(W), maxX = new Int32Array(maxLabel + 1).fill(-1);
    const ysum = new Float64Array(maxLabel + 1), xsum = new Float64Array(maxLabel + 1);
    const cnt = new Int32Array(maxLabel + 1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const l = labels[y * W + x];
      if (l === 0) continue;
      if (y < minY[l]) minY[l] = y; if (y > maxY[l]) maxY[l] = y;
      if (x < minX[l]) minX[l] = x; if (x > maxX[l]) maxX[l] = x;
      ysum[l] += y; xsum[l] += x; cnt[l]++;
    }
    const best = new Float64Array(maxLabel + 1).fill(Infinity);
    const med = new Int32Array(maxLabel + 1).fill(-1);
    const niter = new Int32Array(maxLabel + 1);
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) {
      const i = y * W + x, l = labels[i];
      if (l === 0 || !keep[l] || cnt[l] === 0) continue;
      const dd = (x - xsum[l] / cnt[l]) ** 2 + (y - ysum[l] / cnt[l]) ** 2;
      if (dd < best[l]) { best[l] = dd; med[l] = i; }   // strict <: first in raster order wins
    }
    let maxNiter = 0;
    for (let l = 1; l <= maxLabel; l++) {
      if (!keep[l] || maxY[l] < 0) continue;
      const ly = (maxY[l] - minY[l] + 1) + 2, lx = (maxX[l] - minX[l] + 1) + 2;
      niter[l] = 2 * (ly + lx);
      if (niter[l] > maxNiter) maxNiter = niter[l];
    }
    if (maxNiter === 0) return new Float64Array(maxLabel + 1);

    const S = GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_DST | GPUBufferUsage.COPY_SRC;
    const labBuf = d.createBuffer({ size: HW * 4, usage: S });
    const nitBuf = d.createBuffer({ size: Math.max(256, (maxLabel + 1) * 4), usage: S });
    const medBuf = d.createBuffer({ size: Math.max(256, (maxLabel + 1) * 4), usage: S });
    const T0 = d.createBuffer({ size: HW * 4, usage: S });
    const T1 = d.createBuffer({ size: HW * 4, usage: S });
    const gradBuf = d.createBuffer({ size: 2 * HW * 4, usage: S });
    d.queue.writeBuffer(labBuf, 0, labels instanceof Int32Array ? labels : new Int32Array(labels));
    d.queue.writeBuffer(nitBuf, 0, niter);
    d.queue.writeBuffer(medBuf, 0, med);
    d.queue.writeBuffer(T0, 0, new Float32Array(HW));

    // One uniform buffer holding every iteration's `t`, addressed by dynamic offset.
    const align = d.limits.minUniformBufferOffsetAlignment || 256;
    const uni = d.createBuffer({
      size: align * maxNiter, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST,
    });
    const slot = new Uint32Array(align / 4);   // struct P { H:u32, W:u32, t:i32, _p:u32 }
    for (let t = 0; t < maxNiter; t++) {
      slot[0] = H; slot[1] = W; slot[2] = t; slot[3] = 0;
      d.queue.writeBuffer(uni, t * align, slot);
    }

    const bg = (Tin, Tout) => d.createBindGroup({
      layout: this._flowLayout, entries: [
        { binding: 0, resource: { buffer: uni, size: 16 } },
        { binding: 1, resource: { buffer: labBuf } },
        { binding: 2, resource: { buffer: nitBuf } },
        { binding: 3, resource: { buffer: medBuf } },
        { binding: 4, resource: { buffer: Tin } },
        { binding: 5, resource: { buffer: Tout } },
      ],
    });
    const bgA = bg(T0, T1), bgB = bg(T1, T0);
    const gx = Math.ceil(W / 16), gy = Math.ceil(H / 16);

    const enc = this._mkEncoder("flowqc");
    for (let t = 0; t < maxNiter; t++) {
      const pass = enc.beginComputePass({ label: `flowdiff|${H}x${W}|t${t}` });
      pass.setPipeline(this.pFlowDiff);
      pass.setBindGroup(0, t % 2 === 0 ? bgA : bgB, [t * align]);
      pass.dispatchWorkgroups(gx, gy); pass.end();
    }
    // Iteration t writes T1 for even t, T0 for odd — so the last write lands in T1 when
    // maxNiter is odd and T0 when it is even.
    const finalT = (maxNiter % 2 === 1) ? T1 : T0;
    const gpass = enc.beginComputePass({ label: `flowgrad|${H}x${W}|n${maxNiter}` });
    gpass.setPipeline(this.pFlowGrad);
    gpass.setBindGroup(0, bg(finalT, gradBuf), [0]);
    gpass.dispatchWorkgroups(gx, gy); gpass.end();

    const rb = d.createBuffer({ size: 2 * HW * 4, usage: GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ });
    enc.copyBufferToBuffer(gradBuf, 0, rb, 0, 2 * HW * 4);
    d.queue.submit([enc.finish()]);
    await rb.mapAsync(GPUMapMode.READ);
    const grad = new Float32Array(rb.getMappedRange().slice(0));
    rb.unmap();
    for (const b of [labBuf, nitBuf, medBuf, T0, T1, gradBuf, uni, rb]) b.destroy();

    // Unit-normalise per pixel and accumulate the per-mask error, both in f64 to match
    // the CPU path exactly at the point where the threshold is applied.
    const errSum = new Float64Array(maxLabel + 1), errN = new Int32Array(maxLabel + 1);
    for (let i = 0; i < HW; i++) {
      const l = labels[i];
      if (l === 0 || !keep[l]) continue;
      const gdy = grad[i], gdx = grad[HW + i];
      const mag = Math.sqrt(gdy * gdy + gdx * gdx) + 1e-60;
      const ed = gdy / mag - dy[i], ex = gdx / mag - dx[i];
      errSum[l] += ed * ed + ex * ex; errN[l]++;
    }
    const merr = new Float64Array(maxLabel + 1);
    for (let l = 1; l <= maxLabel; l++) if (errN[l] > 0) merr[l] = errSum[l] / errN[l];
    return merr;
  }

  // Port of cellpose compute_masks post-processing order: remove oversized masks (>40%
  // of image area), then flow-consistency QC (remove_bad_flow_masks), then min-size —
  // matching get_masks_torch -> remove_bad_flow_masks -> fill_holes_and_remove_small_masks.
  // Everything before the flow-QC step, which is where the two paths diverge. Returns
  // the state the QC needs, or null when there is nothing left to filter.
  _filterPrepare(labels, H, W) {
    const maxLabel = labels.reduce((m, v) => v > m ? v : m, 0);
    if (maxLabel === 0) return null;
    const counts = new Int32Array(maxLabel + 1);
    for (let i = 0; i < labels.length; i++) counts[labels[i]]++;
    const big = 0.4 * H * W;
    const keep = new Uint8Array(maxLabel + 1);
    for (let l = 1; l <= maxLabel; l++) keep[l] = (counts[l] > 0 && counts[l] <= big) ? 1 : 0;
    for (let i = 0; i < labels.length; i++) { const l = labels[i]; if (l && !keep[l]) labels[i] = 0; }
    return { maxLabel, counts, keep };
  }

  // Synchronous filtering with the CPU flow reconstruction. Kept for computeMasks(),
  // the non-GPU entry point, and as the reference the GPU path is checked against.
  _filterMasks(labels, dy, dx, H, W, min_size, flow_threshold) {
    const prep = this._filterPrepare(labels, H, W);
    if (!prep) return labels;
    const { maxLabel, keep } = prep;
    const merr = (flow_threshold != null && flow_threshold > 0)
      ? this._maskFlowErrors(labels, dy, dx, H, W, maxLabel, keep) : null;
    return this._filterFinish(labels, prep, min_size, flow_threshold, merr);
  }

  // Same, with the diffusion on the GPU.
  async _filterMasksGPU(labels, dy, dx, H, W, min_size, flow_threshold) {
    const prep = this._filterPrepare(labels, H, W);
    if (!prep) return labels;
    const { maxLabel, keep } = prep;
    let merr = null;
    if (flow_threshold != null && flow_threshold > 0) {
      const mFlow = this._mark();
      merr = await this._maskFlowErrorsGPU(labels, dy, dx, H, W, maxLabel, keep);
      this._stage("gm_flowerr", mFlow);
    }
    return this._filterFinish(labels, prep, min_size, flow_threshold, merr);
  }

  _filterFinish(labels, { maxLabel, counts, keep }, min_size, flow_threshold, merr) {
    if (merr) {
      for (let l = 1; l <= maxLabel; l++) if (keep[l] && merr[l] > flow_threshold) keep[l] = 0;
      for (let i = 0; i < labels.length; i++) { const l = labels[i]; if (l && !keep[l]) labels[i] = 0; }
    }

    // min-size filter + compact renumber
    const remap = new Int32Array(maxLabel + 1);
    let next = 1;
    for (let l = 1; l <= maxLabel; l++) {
      if (!keep[l] || counts[l] < min_size) continue;
      remap[l] = next++;
    }
    for (let i = 0; i < labels.length; i++) labels[i] = remap[labels[i]];
    return labels;
  }

  // full pipeline from a grayscale float image (H,W). Pads to /16 internally.
  // diameter (opts, default 30 = cyto3's trained diam_mean): the image is rescaled by
  // 30/diameter before the forward pass (so cells are ~30px, matching training), then the
  // flow/cellprob fields are resized back to (H,W) before dynamics run at full resolution
  // — mirrors cellpose's default resample=True eval() path.
  async segmentImage(gray, H, W, opts = {}) {
    const t0 = now();
    // Phase 0 stage accounting. Set `_stages` to {} before calling to have the finer
    // CPU/GPU/stall split filled in; leave it null and nothing here costs anything.
    if (this._stages) for (const k of Object.keys(this._stages)) delete this._stages[k];
    const { diameter = 30, chan2 = null } = opts;
    const rescale = 30 / (diameter > 0 ? diameter : 30);
    // `gray` is the channel to segment (cytoplasm); `opts.chan2` is the optional
    // nuclear channel, same [H,W] layout. Each is normalized independently, then
    // both follow the identical resize path so they stay pixel-aligned.
    let H2 = H, W2 = W, ch0 = normalize99(gray);
    let ch1 = chan2 ? normalize99(chan2) : null;
    if (Math.abs(rescale - 1) > 1e-6) {
      H2 = Math.max(1, Math.trunc(H * rescale));   // cellpose uses int()-truncation (transforms.resize_image)
      W2 = Math.max(1, Math.trunc(W * rescale));
      // Cap the working resolution. With tiled inference the forward pass never
      // allocates more than one 224×224 tile's worth of GPU buffers regardless of
      // image size, so the GPU no longer constrains the working resolution (that
      // was the old whole-image path's limit, tied to the decoder's full-res 64-ch
      // buffers). What remains to bound is runtime — the tile count grows with the
      // working resolution — and the host-side blend/dynamics arrays that scale
      // with it. A too-small diameter must not spawn a runaway number of tiles;
      // segmentation still runs, just at a slightly reduced internal resolution.
      const maxPixels = 3_000_000; // ~5x the largest tested sample; a runtime/host-memory bound, not a GPU-buffer one
      if (H2 * W2 > maxPixels) {
        const shrink = Math.sqrt(maxPixels / (H2 * W2));
        const clampedH2 = Math.max(1, Math.round(H2 * shrink)), clampedW2 = Math.max(1, Math.round(W2 * shrink));
        console.warn(`segmentImage: diameter=${diameter} requests a ${H2}x${W2} working resolution, ` +
          `exceeding the ${maxPixels.toLocaleString()}-pixel working-resolution cap; clamped to ${clampedH2}x${clampedW2}.`);
        H2 = clampedH2; W2 = clampedW2;
      }
      ch0 = resizeBilinear(ch0, H, W, H2, W2);
      if (ch1) ch1 = resizeBilinear(ch1, H, W, H2, W2);
    }
    // cellpose's default resample path integrates dynamics at full (H,W) resolution but on
    // flows computed at the rescaled resolution, so it needs niter/rescale iterations
    // (models.py: niter = 200/rescale). Use the *effective* linear rescale actually applied
    // (H2,W2 may be smaller than requested after the clamp above), so large diameters — or a
    // clamped tiny diameter — still get enough iterations to converge.
    const effRescale = Math.sqrt((H2 * W2) / (H * W));
    const niter = Math.max(1, Math.round((opts.niter ?? 200) / effRescale));
    const dynOpts = { ...opts, niter };
    const t1 = now();
    this._stage("preprocess", t0);
    // tiled forward, matching cellpose core.run_net exactly (pad → 224-tiles →
    // per-tile style → taper-blend → crop). Returns [3,H2,W2] = (dY,dX,cellprob).
    const output = await this.runNet(ch0, ch1, H2, W2);
    const t2 = now();
    // dP2 = [dY,dX] (contiguous first two channels), cellprob2 = channel 3
    const dP2 = output.subarray(0, 2 * H2 * W2);
    const cellprob2 = output.subarray(2 * H2 * W2, 3 * H2 * W2);
    // resize flow + cellprob back to the original (H,W) resolution, then run dynamics there
    const mResize = this._mark();
    let dP = dP2, cellprob = cellprob2;
    if (H2 !== H || W2 !== W) {
      dP = new Float32Array(2 * H * W);
      dP.set(resizeBilinear(dP2.subarray(0, H2 * W2), H2, W2, H, W), 0);
      dP.set(resizeBilinear(dP2.subarray(H2 * W2, 2 * H2 * W2), H2, W2, H, W), H * W);
      cellprob = resizeBilinear(cellprob2, H2, W2, H, W);
    }
    this._stage("resize_back", mResize);
    const labels = await this.computeMasksGPU(dP, cellprob, H, W, dynOpts);
    const t3 = now();
    // The three coarse buckets stay exactly as they were — registry.js and the Python
    // shims read them — with the finer split carried alongside when profiling is on.
    const timings = { preprocess: t1 - t0, forward: t2 - t1, dynamics: t3 - t2, total: t3 - t0 };
    if (this._stages) timings.stages = { ...this._stages };
    return { labels, output, dP, cellprob, H, W, timings };
  }
}

// ---- preprocessing (mirror of baseline_pytorch.py) ----
// Percentile-normalize one channel to [0,1] over its 1st..99th percentile.
//
// The two guards mirror cellpose's own behaviour (transforms.normalize_img's
// per-channel `np.ptp(...) > 0` test, plus transforms.normalize99's
// `x99 - x01 > 1e-3` test) and matter as soon as a caller supplies a real
// second (nuclear) channel: an empty or near-constant channel is entirely
// plausible input — a user picking the wrong channel, or an image whose
// nuclear plane is blank — and dividing by its ~0 spread would otherwise
// explode a dead channel into garbage that the net happily consumes.
//   - ptp == 0 (perfectly constant): left untouched, exactly as cellpose does.
//   - spread <= 1e-3: zeroed.
// For real grayscale neither fires, so the validated single-channel path is
// bit-for-bit unchanged.
export function normalize99(x) {
  let mn = Infinity, mx = -Infinity;
  for (let i = 0; i < x.length; i++) { const v = x[i]; if (v < mn) mn = v; if (v > mx) mx = v; }
  const out = new Float32Array(x.length);
  if (!(mx > mn)) { out.set(x); return out; }   // constant channel: cellpose leaves it as-is
  const s = Float32Array.from(x).sort();
  const p = (q) => s[Math.min(s.length - 1, Math.max(0, Math.floor(q / 100 * (s.length - 1))))];
  const p1 = p(1), p99 = p(99);
  if (p99 - p1 <= 1e-3) return out;             // no usable dynamic range: cellpose zeroes it
  const denom = (p99 - p1) + 1e-10;
  for (let i = 0; i < x.length; i++) out[i] = (x[i] - p1) / denom;
  return out;
}
// bilinear resize, pixel-center sampling (matches cv2.resize/INTER_LINEAR, what
// cellpose's transforms.resize_image uses for diameter rescale + un-resizing flows).
export function resizeBilinear(src, sh, sw, dh, dw) {
  const out = new Float32Array(dh * dw);
  if (sh === dh && sw === dw) { out.set(src); return out; }
  const sy = sh / dh, sx = sw / dw;
  for (let y = 0; y < dh; y++) {
    let fy = (y + 0.5) * sy - 0.5;
    if (fy < 0) fy = 0; else if (fy > sh - 1) fy = sh - 1;
    const y0 = fy | 0, y1 = y0 < sh - 1 ? y0 + 1 : y0, wy = fy - y0;
    for (let x = 0; x < dw; x++) {
      let fx = (x + 0.5) * sx - 0.5;
      if (fx < 0) fx = 0; else if (fx > sw - 1) fx = sw - 1;
      const x0 = fx | 0, x1 = x0 < sw - 1 ? x0 + 1 : x0, wx = fx - x0;
      const v00 = src[y0 * sw + x0], v01 = src[y0 * sw + x1];
      const v10 = src[y1 * sw + x0], v11 = src[y1 * sw + x1];
      out[y * dw + x] = v00 * (1 - wy) * (1 - wx) + v01 * (1 - wy) * wx +
                         v10 * wy * (1 - wx) + v11 * wy * wx;
    }
  }
  return out;
}
// ---- tiled inference (exact mirror of cellpose core.run_net) ----------------
// cellpose never runs an image larger than 224px (either dim) through the net in
// one pass. It zero-pads the working-resolution image, cuts it into overlapping
// 224×224 tiles, runs the net *per tile* (so the style vector that conditions the
// decoder is computed from each tile's own local view — the crux of the fidelity
// gap), then blends the per-tile outputs with a tapered weight mask. These helpers
// reproduce transforms.get_pad_yx / make_tiles / _taper_mask / average_tiles and
// core.run_net's pad-tile-blend-crop exactly (validated against tiling_ref.py).

// transforms.get_pad_yx(Ly,Lx,div=16,extra=1): pads to a multiple of `div` PLUS an
// extra `extra*div//2` on every side. Returns [ypad1,ypad2,xpad1,xpad2].
export function getPadYX(Ly, Lx, div = 16, extra = 1) {
  const half = Math.floor((extra * div) / 2);
  const padAxis = (L) => {
    const Lpad = Math.ceil(L / div) * div - L;
    return [half + Math.floor(Lpad / 2), half + Lpad - Math.floor(Lpad / 2)];
  };
  const [ypad1, ypad2] = padAxis(Ly);
  const [xpad1, xpad2] = padAxis(Lx);
  return [ypad1, ypad2, xpad1, xpad2];
}

// np.linspace(0, stop, n).astype(int) — evenly spaced tile origins, truncated to int.
function linspaceInt(stop, n) {
  if (n <= 1) return [0];
  const step = stop / (n - 1), out = new Array(n);
  for (let k = 0; k < n; k++) out[k] = Math.trunc(step * k);
  return out;
}

// transforms.make_tiles' non-augment grid: tile count + top-left corners, in the
// PADDED image's coordinates. `Ly`/`Lx` are the padded dims.
export function tileGrid(Ly, Lx, bsize = 224, tile_overlap = 0.1) {
  const ov = Math.min(0.5, Math.max(0.05, tile_overlap));
  const bsizeY = Math.min(bsize, Ly), bsizeX = Math.min(bsize, Lx);
  const ny = Ly <= bsize ? 1 : Math.ceil(((1 + 2 * ov) * Ly) / bsize);
  const nx = Lx <= bsize ? 1 : Math.ceil(((1 + 2 * ov) * Lx) / bsize);
  return { ny, nx, bsizeY, bsizeX,
           ystart: linspaceInt(Ly - bsizeY, ny),
           xstart: linspaceInt(Lx - bsizeX, nx) };
}

// transforms._taper_mask(ly,lx): separable logistic falloff, full-strength at tile
// center, ~0 at edges, so overlapping tiles blend smoothly. Returns Float64Array
// [ly*lx]; mask2d[i,j] = m1d[i]*m1d[j].
export function taperMask(ly, lx, sig = 7.5) {
  const bsize = Math.max(224, Math.max(ly, lx));
  const mean = (bsize - 1) / 2, edge = bsize / 2 - 20;
  const m1d = new Float64Array(bsize);
  for (let i = 0; i < bsize; i++)
    m1d[i] = 1 / (1 + Math.exp((Math.abs(i - mean) - edge) / sig));
  const y0 = Math.floor(bsize / 2) - Math.floor(ly / 2);
  const x0 = Math.floor(bsize / 2) - Math.floor(lx / 2);
  const mask = new Float64Array(ly * lx);
  for (let a = 0; a < ly; a++) for (let b = 0; b < lx; b++)
    mask[a * lx + b] = m1d[y0 + a] * m1d[x0 + b];
  return mask;
}

// Pad already-normalized channels [H,W] to a multiple of 16 (reflect edge) and
// return the 2-channel net input [2,Hp,Wp].
//
// cyto3's first layer always takes 2 channels: [cytoplasm, nucleus]. Passing
// ch1 = null leaves the nuclear plane zero-filled, which is exactly what
// cellpose does for grayscale input (channels=[0,0]); passing a real ch1 is the
// channels=[2,3]-style two-channel path.
function padTo16(ch0, ch1, H, W) {
  const div = 16;
  const Hp = Math.ceil(H / div) * div, Wp = Math.ceil(W / div) * div;
  const data = new Float32Array(2 * Hp * Wp); // ch1 stays 0 when not supplied
  const planes = [ch0, ch1];
  for (let c = 0; c < 2; c++) {
    const src = planes[c];
    if (!src) continue;
    const off = c * Hp * Wp;
    for (let y = 0; y < H; y++) for (let x = 0; x < W; x++) data[off + y * Wp + x] = src[y * W + x];
    // Edge-replicate the last row, then the last column — bit-for-bit what the
    // PyTorch reference does (baseline_pytorch.py `pad_to`: the `x[:,H-1:H,:]`
    // slice is size-1 on the row axis and broadcasts across every pad row, so
    // despite the "reflect" intent it replicates the edge). The port validates
    // against that reference (test_forward.js preprocess check), so it must match
    // it exactly; the column pass reads col W-1 of each row *after* row padding,
    // matching pad_to's ordering so the bottom-right corner replicates too.
    for (let y = H; y < Hp; y++) for (let x = 0; x < W; x++)
      data[off + y * Wp + x] = data[off + (H - 1) * Wp + x];
    for (let y = 0; y < Hp; y++) for (let x = W; x < Wp; x++)
      data[off + y * Wp + x] = data[off + y * Wp + (W - 1)];
  }
  return { data, Hp, Wp };
}
/**
 * Raw channels -> padded 2-channel net input. `chan2` (the nuclear channel) is
 * optional; pass null for cellpose's grayscale mode. Each channel is normalized
 * independently, as cellpose normalizes per channel.
 */
export function preprocess(chan, chan2, H, W) {
  return padTo16(normalize99(chan), chan2 ? normalize99(chan2) : null, H, W);
}
export function preprocessGray(gray, H, W) {
  return preprocess(gray, null, H, W);
}

function now() {
  return (typeof performance !== "undefined" ? performance.now() : Date.now());
}
