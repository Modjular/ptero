// Adapter and device acquisition, shared by the three model modules.
//
// This exists because the limits are load-bearing, not as a convenience. The conv kernel
// stages several input channels per barrier round and needs more threadgroup memory than
// WebGPU's 16 KB default allows; a device without the raised limit fails to build the
// pipeline outright. Duplicating that reasoning in three `create()` methods was the
// alternative, and it would go stale.
//
// Each model module still owns its own device — nothing here is a singleton — so the
// "one self-contained ES module per model" property in the README is unchanged. A host
// that wants one device across several models passes it to `load({ device })`.

/**
 * Request an adapter and a device with everything the kernels need.
 *
 * Features are requested only when the adapter reports them, because requestDevice()
 * rejects outright on an unavailable requiredFeature rather than degrading:
 *
 *   timestamp-query  per-dispatch GPU timing, used by src/profile/. Costs nothing when
 *                    unused — the query sets are only created by a profiling run.
 *   shader-f16       half-width shared memory. Currently only exercised by a benchmark
 *                    variant (see docs/PHASE1.md on why f16 was not adopted).
 */
export async function requestDevice({ powerPreference = "high-performance" } = {}) {
  if (!navigator.gpu) throw new Error("no WebGPU — use Chrome/Edge 113+, Safari 18+, or deno --unstable-webgpu");
  const adapter = await navigator.gpu.requestAdapter({ powerPreference });
  if (!adapter) throw new Error("no WebGPU adapter");
  const lim = adapter.limits;
  const device = await adapter.requestDevice({
    requiredFeatures: ["timestamp-query", "shader-f16"].filter((f) => adapter.features.has(f)),
    requiredLimits: {
      maxBufferSize: lim.maxBufferSize,
      maxStorageBufferBindingSize: lim.maxStorageBufferBindingSize,
      maxComputeInvocationsPerWorkgroup: lim.maxComputeInvocationsPerWorkgroup,
      // Required, not optional: the conv kernel's input-channel staging exceeds the
      // 16 KB default. A limit is a ceiling, not an allocation, so asking for the
      // adapter's maximum costs nothing.
      maxComputeWorkgroupStorageSize: lim.maxComputeWorkgroupStorageSize,
    },
  });
  device.__adapterInfo = adapter.info ?? null;
  return device;
}

/** Best-effort GPU name for diagnostics. Adapter info is deliberately vague in browsers. */
export function adapterDescription(device) {
  const i = device?.__adapterInfo;
  if (!i) return "unknown GPU";
  return [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" ") || "unknown GPU";
}
