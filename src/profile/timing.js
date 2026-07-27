// Per-dispatch GPU timing for the segmentation engines.
//
// WebGPU gives us exactly one performance primitive: `timestamp-query`, written at
// compute-pass boundaries. No DRAM-traffic counters, no occupancy, no equivalent of
// Nsight's priority score. Everything else in Phase 0 is reconstructed analytically
// (see cost.js), which is only possible because every kernel in this repo is
// hand-written and its ideal traffic is therefore exactly known.
//
// The engines are near-frozen, so this attaches by *substitution* rather than by
// editing the op methods: each engine builds its command encoders through a
// `_mkEncoder` hook that defaults to `device.createCommandEncoder`. Swap the hook for
// `recorder.wrap(...)` and every `beginComputePass()` underneath silently acquires
// `timestampWrites`, with no branch anywhere on the unprofiled path.
//
//   const rec = new DispatchRecorder(device);
//   cp._mkEncoder = (label) => rec.wrap(device.createCommandEncoder({ label }), label);
//   await cp.segmentImage(...);
//   const records = await rec.drain();   // [{ label, group, ns, workgroups, ... }]
//
// Timestamps are nanoseconds. Verified unquantized on Chrome + Metal with the flags
// tools/drive.mjs already passes — Chrome's 100 µs timestamp quantization, which would
// have made per-dispatch measurement useless, is not applied there. If you ever see
// every duration land on a multiple of 100000, that is what has changed.

const RESOLVE_USAGE = GPUBufferUsage.QUERY_RESOLVE | GPUBufferUsage.COPY_SRC;
const READ_USAGE = GPUBufferUsage.COPY_DST | GPUBufferUsage.MAP_READ;

export class DispatchRecorder {
  /**
   * @param device   the shared GPUDevice
   * @param capacity max compute passes per command encoder. One CPnet tile forward is
   *                 ~61 passes; the ceiling exists so a runaway encoder can't blow past
   *                 the 4096-entry GPUQuerySet limit silently.
   */
  constructor(device, { capacity = 512 } = {}) {
    this.device = device;
    this.capacity = capacity;
    this.available = !!device.features?.has?.("timestamp-query");
    this.groups = [];      // encoders submitted but not yet read back
    this.records = [];      // drained, in submission order
    this.overflowed = 0;    // passes that exceeded `capacity` and went untimed
  }

  /**
   * Wrap a GPUCommandEncoder so its compute passes are timed. Returns something that
   * behaves exactly like the encoder — the engine cannot tell the difference, and if
   * timestamps are unavailable it *is* the encoder, unwrapped.
   */
  wrap(encoder, group = "") {
    if (!this.available) return encoder;
    const d = this.device;
    const qs = d.createQuerySet({ type: "timestamp", count: this.capacity * 2 });
    const g = { group, qs, passes: [], resolve: null, read: null };
    this.groups.push(g);

    const proxy = new Proxy(encoder, {
      get: (target, prop, receiver) => {
        if (prop === "beginComputePass") {
          return (desc = {}) => {
            const i = g.passes.length;
            if (i >= this.capacity) {
              this.overflowed++;
              return target.beginComputePass(desc);
            }
            const entry = { label: desc.label ?? `pass${i}`, workgroups: null };
            g.passes.push(entry);
            const pass = target.beginComputePass({
              ...desc,
              timestampWrites: {
                querySet: qs,
                beginningOfPassWriteIndex: i * 2,
                endOfPassWriteIndex: i * 2 + 1,
              },
            });
            // Proxy the pass too, purely to capture the dispatch geometry. Recording it
            // here rather than recomputing it in cost.js means the wave-quantization
            // numbers are the grid the GPU actually saw, not a second derivation of it
            // that could drift from the engine.
            return new Proxy(pass, {
              get: (pt, pp, pr) => {
                if (pp === "dispatchWorkgroups") {
                  return (x, y = 1, z = 1) => {
                    entry.workgroups = [x, y, z];
                    return pt.dispatchWorkgroups(x, y, z);
                  };
                }
                const v = Reflect.get(pt, pp, pr);
                return typeof v === "function" ? v.bind(pt) : v;
              },
            });
          };
        }
        if (prop === "finish") {
          // The resolve has to be recorded into this same encoder, and finish() is by
          // construction the last thing anyone calls on it — so this is the one safe
          // place to append it without the engine having to know we exist.
          return (...a) => {
            const n = g.passes.length;
            if (n > 0) {
              g.resolve = d.createBuffer({ size: n * 2 * 8, usage: RESOLVE_USAGE });
              g.read = d.createBuffer({ size: n * 2 * 8, usage: READ_USAGE });
              target.resolveQuerySet(qs, 0, n * 2, g.resolve, 0);
              target.copyBufferToBuffer(g.resolve, 0, g.read, 0, n * 2 * 8);
            }
            return target.finish(...a);
          };
        }
        const v = Reflect.get(target, prop, receiver);
        return typeof v === "function" ? v.bind(target) : v;
      },
    });
    return proxy;
  }

  /**
   * Read back everything submitted so far. Safe to call after the engine's own
   * readback has resolved — the copy is already complete by then, so mapping costs
   * no additional GPU wait.
   *
   * Each record carries `span`: the wall-clock extent of its whole command encoder on
   * the GPU. Comparing sum(ns) against span is the overlap check — Metal is free to run
   * compute passes concurrently, and if it does, per-pass attribution is not sound.
   */
  async drain() {
    const out = [];
    for (const g of this.groups) {
      if (!g.read) { g.qs.destroy?.(); continue; }
      await g.read.mapAsync(GPUMapMode.READ);
      const t = new BigUint64Array(g.read.getMappedRange().slice(0));
      g.read.unmap();
      let lo = null, hi = null;
      const rows = [];
      for (let i = 0; i < g.passes.length; i++) {
        const b = t[i * 2], e = t[i * 2 + 1];
        // A query that never got written reads back as zero. Skip rather than
        // reporting a fabricated zero-length dispatch.
        if (b === 0n && e === 0n) continue;
        if (lo === null || b < lo) lo = b;
        if (hi === null || e > hi) hi = e;
        rows.push({ ...g.passes[i], group: g.group, ns: Number(e - b) });
      }
      const span = lo === null ? 0 : Number(hi - lo);
      for (const r of rows) out.push({ ...r, span });
      g.read.destroy(); g.resolve.destroy(); g.qs.destroy?.();
    }
    this.groups = [];
    this.records.push(...out);
    return out;
  }

  /**
   * Sum of per-pass durations against the GPU span of each encoder. Ratios near 1 mean
   * passes ran back-to-back and per-pass numbers can be trusted; ratios well above 1
   * mean passes overlapped and only the span is meaningful.
   */
  overlapReport() {
    const byGroup = new Map();
    for (const r of this.records) {
      // `span` is a property of the encoder, so key on it alongside the group name.
      const k = r.group + "#" + r.span;
      const e = byGroup.get(k) || { group: r.group, span: r.span, sum: 0, passes: 0 };
      e.sum += r.ns; e.passes++;
      byGroup.set(k, e);
    }
    const rows = [...byGroup.values()].filter((e) => e.span > 0);
    const ratios = rows.map((e) => e.sum / e.span).sort((a, b) => a - b);
    return {
      encoders: rows.length,
      medianRatio: ratios.length ? ratios[ratios.length >> 1] : null,
      maxRatio: ratios.length ? ratios[ratios.length - 1] : null,
      serial: ratios.length ? ratios[ratios.length - 1] <= 1.15 : null,
    };
  }

  reset() { this.records = []; this.overflowed = 0; }
}

/**
 * Attach a recorder to any engine exposing the `_mkEncoder` hook. Returns a detach
 * function so a profiling run can leave the engine exactly as it found it.
 */
export function attach(engine, recorder) {
  const original = engine._mkEncoder;
  engine._mkEncoder = (label) =>
    recorder.wrap(recorder.device.createCommandEncoder({ label }), label);
  return () => { engine._mkEncoder = original; };
}

/**
 * Cost of an empty dispatch — the floor below which a duration says nothing about the
 * work, only about launch overhead. This is the threshold for classify.js's
 * "launch-limited" bucket, and it is measured rather than assumed because it is a
 * property of this browser on this driver on this GPU.
 */
export async function launchFloor(device, { reps = 64 } = {}) {
  if (!device.features?.has?.("timestamp-query")) return null;
  const mod = device.createShaderModule({
    code: `@compute @workgroup_size(1) fn main() {}`,
  });
  const pipe = device.createComputePipeline({
    layout: "auto", compute: { module: mod, entryPoint: "main" },
  });
  const qs = device.createQuerySet({ type: "timestamp", count: reps * 2 });
  const res = device.createBuffer({ size: reps * 2 * 8, usage: RESOLVE_USAGE });
  const rb = device.createBuffer({ size: reps * 2 * 8, usage: READ_USAGE });
  const enc = device.createCommandEncoder({ label: "launch-floor" });
  for (let i = 0; i < reps; i++) {
    const p = enc.beginComputePass({
      timestampWrites: {
        querySet: qs, beginningOfPassWriteIndex: i * 2, endOfPassWriteIndex: i * 2 + 1,
      },
    });
    p.setPipeline(pipe); p.dispatchWorkgroups(1); p.end();
  }
  enc.resolveQuerySet(qs, 0, reps * 2, res, 0);
  enc.copyBufferToBuffer(res, 0, rb, 0, reps * 2 * 8);
  device.queue.submit([enc.finish()]);
  await rb.mapAsync(GPUMapMode.READ);
  const t = new BigUint64Array(rb.getMappedRange().slice(0));
  rb.unmap();
  const ns = [];
  for (let i = 0; i < reps; i++) ns.push(Number(t[i * 2 + 1] - t[i * 2]));
  ns.sort((a, b) => a - b);
  rb.destroy(); res.destroy(); qs.destroy?.();
  // Median, not mean: the first dispatch pays for pipeline warm-up and would otherwise
  // drag the floor upward.
  return { medianNs: ns[ns.length >> 1], minNs: ns[0], maxNs: ns[ns.length - 1] };
}
