// The model registry — the one place that knows what models exist, where their
// weights live, what input shape each wants, and what each is actually good for.
//
// Everything above this layer (the Python shims, the notebook, the agent) addresses
// models by string id and hands over an image; the per-engine differences in input
// layout, option names and return shape are reconciled here and nowhere else.
//
// Weights are fetched lazily, per model, on first use and cached forever after — a
// session that only ever runs StarDist never downloads cyto3's 26 MB.
// The engines are vendored from webgpu-cellseg — see vendor/webgpu-cellseg/VENDOR.md.
// Don't edit them here; changes go upstream and come back through a re-sync.
import { CellposeWebGPU } from "../vendor/webgpu-cellseg/src/cellpose.js";
import { StarDistWebGPU } from "../vendor/webgpu-cellseg/src/stardist.js";
import { InstanSegWebGPU } from "../vendor/webgpu-cellseg/src/instanseg.js";
import { sharedDevice, adapterDescription } from "./gpu.js";

// `input` is what the engine's segmentImage() wants:
//   "gray" — one [H,W] Float32Array   ("chan2: true" = accepts an optional second one)
//   "rgb"  — three planes, [3,H,W]
// `params` are the engine's own defaults, lifted from the upstream webgpu-cellseg demo
// pages and engine source rather than invented, so a call with no options behaves
// exactly like the demo does:
//   stardist thresholds  — stardist demo MODELS
//   instanseg thresholds — src/instanseg.js DEFAULTS
//   cellpose thresholds  — src/cellpose.js computeMasksGPU
export const MODELS = {
  "cellpose-cyto3": {
    label: "Cellpose cyto3",
    engine: CellposeWebGPU,
    dir: "cellpose-cyto3",
    input: "gray",
    chan2: true,
    mb: 26,
    // Raw weights.bin is ~25.2 MiB, just over Cloudflare's 25 MiB per-asset limit.
    // weightsGz points at a gzip-compressed weights.bin.gz instead; getModel()
    // decompresses it client-side with DecompressionStream before handing the
    // ArrayBuffer to loadWeights(). Every other model's weights fit under the limit
    // as plain weights.bin and don't need this.
    weightsGz: true,
    good_for: "generalist cells and cytoplasm, fluorescence or brightfield; also finds " +
              "nuclei when given the nuclear channel at a smaller diameter",
    key_param: "diameter — the approximate object size in pixels; the single most " +
               "important knob. 30 is the trained default; pass the real cell size.",
    params: { diameter: 30, cellprob_threshold: 0.0, flow_threshold: 0.4, min_size: 15, niter: 200 },
  },
  "stardist-fluo": {
    label: "StarDist 2D_versatile_fluo",
    engine: StarDistWebGPU,
    dir: "stardist-fluo",
    input: "gray",
    mb: 5.7,
    good_for: "fluorescent nuclei (DAPI / Hoechst) — the right choice for a nuclear " +
              "stain channel; it is a nuclei model and does poorly on cytoplasm",
    key_param: "prob_thresh — lower to find more (dimmer) nuclei, raise to find fewer.",
    params: { prob_thresh: 0.479, nms_thresh: 0.3 },
  },
  "stardist-he": {
    label: "StarDist 2D_versatile_he",
    engine: StarDistWebGPU,
    dir: "stardist-he",
    input: "rgb",
    mb: 5.7,
    good_for: "H&E histology nuclei — trained on RGB brightfield tissue scans, not " +
              "fluorescence",
    key_param: "prob_thresh — lower to find more nuclei, raise to find fewer.",
    params: { prob_thresh: 0.6925, nms_thresh: 0.3 },
  },
  "instanseg-brightfield": {
    label: "InstanSeg brightfield_nuclei",
    engine: InstanSegWebGPU,
    dir: "instanseg-brightfield",
    input: "rgb",
    mb: 15,
    good_for: "brightfield / unstained nuclei — phase-contrast or plain transmitted " +
              "light, where there is no fluorescent nuclear stain to threshold",
    key_param: "seed_threshold — lower to detect more nuclei; mask_threshold controls " +
               "how far each detected nucleus grows.",
    params: { seed_threshold: 0.7, mask_threshold: 0.53, min_size: 10 },
  },
};

const WEIGHTS_BASE = new URL("../weights/", import.meta.url).href;
const loading = new Map();   // id -> Promise<engine instance>, in-flight or settled

export function modelIds() { return Object.keys(MODELS); }

// True once a model's weights are resident — lets callers report "already loaded" vs
// "this will cost a 26 MB download" without triggering the download to find out.
export function isResident(id) { return loading.has(id); }

/**
 * Fetch + instantiate a model, or return the already-loaded instance. Concurrent calls
 * for the same id share one in-flight promise, so two cells racing to segment don't
 * download the weights twice.
 */
export function getModel(id) {
  const cfg = MODELS[id];
  if (!cfg) throw new Error(`unknown model "${id}" — known models: ${modelIds().join(", ")}`);
  if (!loading.has(id)) {
    const p = (async () => {
      const device = await sharedDevice();
      const base = WEIGHTS_BASE + cfg.dir + "/";
      if (!cfg.weightsGz) return cfg.engine.load(base, { device });
      // Same two fetches the engines' own convenience load() does, just decompressing
      // weights.bin.gz first — loadWeights() is the same public entry point load()
      // calls internally, so this instantiates identically to the uncompressed path.
      const [manifest, bin] = await Promise.all([
        fetch(base + "manifest.json").then(r => r.json()),
        fetch(base + "weights.bin.gz")
          .then(r => new Response(r.body.pipeThrough(new DecompressionStream("gzip"))).arrayBuffer()),
      ]);
      const inst = new cfg.engine(device);
      inst.loadWeights(manifest, bin);
      return inst;
    })();
    // Don't cache a rejection: a failed fetch (offline, weights not served) must be
    // retryable rather than poisoning this model for the page's lifetime.
    p.catch(() => loading.delete(id));
    loading.set(id, p);
  }
  return loading.get(id);
}

// ---- input adaptation -------------------------------------------------------------
// The three engines disagree about how an image arrives. Callers above this layer use
// one convention — an array of [H,W] Float32Array planes — and this reshapes per model.

function toPlanes(image) {
  if (image instanceof Float32Array) return [image];
  if (Array.isArray(image)) return image.map(p => (p instanceof Float32Array ? p : Float32Array.from(p)));
  throw new Error("image must be a Float32Array or an array of Float32Array planes");
}

// [3,H,W] planar, replicating a single plane across R/G/B — the same rule the upstream
// demo pages use for feeding a grayscale source to an RGB model (instanseg selectRGB).
// Extra planes past the third are dropped; two planes get the second duplicated into B,
// which beats erroring on a 2-channel fluorescence stack.
function asRGB(planes, H, W) {
  const n = H * W, rgb = new Float32Array(3 * n);
  for (let c = 0; c < 3; c++) rgb.set(planes[Math.min(c, planes.length - 1)], c * n);
  return rgb;
}

/**
 * Segment an image with any registered model.
 *
 *   image  a [H,W] Float32Array, or an array of such planes
 *   opts   engine options; unspecified keys fall back to the model's own defaults.
 *          For cellpose-cyto3 a second plane is passed as the optional nuclear
 *          channel (chan2) rather than being collapsed into the first.
 *
 * Returns { labels: Int32Array[H*W], n, timings } — 0 is background, and labels are
 * consecutive from 1, so labels.max() is the object count for every model.
 */
export async function segment(id, image, H, W, opts = {}) {
  const cfg = MODELS[id];
  if (!cfg) throw new Error(`unknown model "${id}" — known models: ${modelIds().join(", ")}`);
  const planes = toPlanes(image);
  for (const p of planes) {
    if (p.length !== H * W) throw new Error(`plane has ${p.length} pixels, expected ${H * W} (${H}×${W})`);
  }
  const model = await getModel(id);
  const merged = { ...cfg.params, ...opts };

  let res;
  if (cfg.input === "rgb") {
    // StarDist's H&E checkpoint wants an array of channels; InstanSeg wants one packed
    // [3,H,W] buffer. Both are "three planes", spelled differently.
    const rgb = asRGB(planes, H, W);
    const n = H * W;
    res = cfg.engine === StarDistWebGPU
      ? await model.segmentImage([rgb.subarray(0, n), rgb.subarray(n, 2 * n), rgb.subarray(2 * n)], H, W, merged)
      : await model.segmentImage(rgb, H, W, merged);
  } else if (cfg.chan2 && planes.length > 1) {
    res = await model.segmentImage(planes[0], H, W, { ...merged, chan2: planes[1] });
  } else {
    res = await model.segmentImage(planes[0], H, W, merged);
  }

  const { labels } = res;
  let n = 0;
  for (let i = 0; i < labels.length; i++) if (labels[i] > n) n = labels[i];
  return { labels, n, timings: res.timings };
}

/**
 * Plain-JSON description of every model, for ptero.models.list() and the agent's
 * system prompt. Deliberately free of engine class references so it survives
 * structuredClone across the worker boundary and JSON.stringify into a prompt.
 */
export function catalogue() {
  return Object.entries(MODELS).map(([id, cfg]) => ({
    id,
    label: cfg.label,
    good_for: cfg.good_for,
    key_param: cfg.key_param,
    input: cfg.input === "rgb" ? "rgb (3 planes; a single plane is replicated)" : "grayscale (1 plane)",
    accepts_second_channel: !!cfg.chan2,
    defaults: { ...cfg.params },
    download_mb: cfg.mb,
    resident: isResident(id),
  }));
}

// Diagnostics for ptero.env — never triggers a device request of its own.
export async function environment() {
  let gpu = "not initialised";
  try { gpu = adapterDescription(await sharedDevice()); } catch (e) { gpu = "unavailable: " + e.message; }
  return { gpu, models: modelIds(), resident: modelIds().filter(isResident) };
}
