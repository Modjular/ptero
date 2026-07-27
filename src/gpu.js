// One GPUDevice for every engine on the page.
//
// Each of the three cores used to call navigator.gpu.requestAdapter() from its own
// static create(), which was fine when a page only ever instantiated one of them (the
// demo pages) but means three adapters and three devices as soon as a single page can
// reach for cellpose, stardist and instanseg — as the notebook now can. Devices are not
// free: each carries its own allocator and its own set of compiled pipelines, and
// buffers are not transferable between them.
//
// All three cores asked for the identical limit set (each at whatever the adapter
// reports as its maximum), so there is nothing to reconcile here — this is the same
// request, made once.
let devicePromise = null;

export function sharedDevice() {
  if (!devicePromise) {
    devicePromise = (async () => {
      if (!navigator.gpu) throw new Error("no WebGPU — use Chrome/Edge, or Safari 18+");
      const adapter = await navigator.gpu.requestAdapter({ powerPreference: "high-performance" });
      if (!adapter) throw new Error("no WebGPU adapter");
      const lim = adapter.limits;
      // `timestamp-query` is what src/profile/ measures per-dispatch GPU time with. It
      // must be feature-detected rather than requested outright — requestDevice()
      // *rejects* if a requiredFeature is unavailable, so asking unconditionally would
      // break every machine that lacks it. It downloads nothing and allocates nothing,
      // so holding it costs the ~5 MB cold start nothing; the query sets themselves are
      // only created when a profiling run asks for them.
      // `shader-f16` is here for the same reason and on the same terms: it lets a kernel
      // hold shared-memory tiles at half width. Both are feature-detected because
      // requestDevice() rejects outright on an unavailable requiredFeature.
      const requiredFeatures = ["timestamp-query", "shader-f16"]
        .filter((f) => adapter.features.has(f));
      const device = await adapter.requestDevice({
        requiredFeatures,
        requiredLimits: {
          maxBufferSize: lim.maxBufferSize,
          maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
          maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
        }
      });
      device.__adapterInfo = adapter.info ?? null;   // surfaced through ptero.env
      // A lost device would otherwise leave every cached engine instance holding a
      // dead handle and failing opaquely. Drop the memo so the next call rebuilds.
      device.lost?.then((info) => {
        console.warn("WebGPU device lost:", info.message);
        devicePromise = null;
      });
      return device;
    })();
    // A failed request must not be cached as a permanently-rejected promise, or a
    // transient failure poisons the page for its whole lifetime.
    devicePromise.catch(() => { devicePromise = null; });
  }
  return devicePromise;
}

// Best-effort human-readable GPU name for ptero.env / diagnostics. Adapter info is
// deliberately vague in browsers (fingerprinting), so treat every field as optional.
export function adapterDescription(device) {
  const i = device?.__adapterInfo;
  if (!i) return "unknown GPU";
  return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" ") || "unknown GPU";
}
