// Phase 0 measurement run, browser side. tools/profile.mjs drives this and does nothing
// but launch Chrome and write the JSON out, so the logic lives here with the rest of the
// engine code rather than inside a template string in the driver.
//
// One run produces everything Phase 0 asks for: the machine's empirical roofs, the
// stage-resolved Amdahl split, the per-dispatch table with analytical costs, the
// time-weighted attainable-roof-normalised roofline, the four-way classification, and
// both gate decisions.

import { sharedDevice, adapterDescription } from "../gpu.js";
import { CellposeWebGPU } from "../cellpose.js";
import { DispatchRecorder, attach, launchFloor } from "./timing.js";
import { measureRoofs } from "./roofs.js";
import { annotate, summarise } from "./cost.js";
import { classify, gates } from "./classify.js";

const median = (xs) => {
  if (!xs.length) return null;
  const s = [...xs].sort((a, b) => a - b);
  return s[s.length >> 1];
};
const spread = (xs) => {
  if (xs.length < 2) return 0;
  const s = [...xs].sort((a, b) => a - b);
  return (s[s.length - 1] - s[0]) / (median(xs) || 1);
};

/** Fetch an image URL and decode it to Float32 planes via the demos' loader. */
async function loadPlanes(url) {
  const { loadSource } = await import("../../demo/tiff-loader.js");
  const name = url.split("/").pop();
  let src;
  if (/\.tiff?$/i.test(name)) {
    // loadSource only takes the TIFF path for a File, so wrap the fetched bytes.
    const buf = await (await fetch(url)).arrayBuffer();
    src = await loadSource(new File([buf], name));
  } else {
    src = await loadSource(url);
  }
  const { W, H } = src;
  if (src.planes) return { W, H, planes: src.planes.map((p) => p.data ?? p) };
  // Ordinary image: imgData is RGBA8. Take luminance-free plane 0 — these samples are
  // greyscale to begin with, and quantising further would change what we measure.
  const d = src.imgData.data;
  const g = new Float32Array(W * H);
  for (let i = 0; i < W * H; i++) g[i] = d[i * 4];
  return { W, H, planes: [g] };
}

/**
 * Profile one workload.
 *
 * `repeats` runs are timed after a discarded warm-up. Per-dispatch durations are
 * reduced element-wise by position — the dispatch sequence is deterministic, so
 * position is a stable key — and the run-to-run spread travels with the result because
 * this is a passively cooled laptop and thermal drift is a real confound.
 */
export async function profileWorkload(cp, wl, { repeats = 5, waveCapacity = null } = {}) {
  let { W, H, planes } = await loadPlanes(wl.url);
  // `crop` exists so the single-tile fast path is reachable at all: none of the sample
  // images fit inside one 224px tile at diameter 30, and that path skips the taper-blend
  // entirely, so without it a whole branch of runNet goes unmeasured.
  //
  // Centred, not top-left. A corner crop of a microscopy field is usually background,
  // and a workload with no cells in it measures the forward pass honestly but makes the
  // mask-assembly stage disappear.
  //
  // Note the size constraint: getPadYX pads to a multiple of 16 and then adds div/2 on
  // every side, so one tile needs ceil(L/16)*16 <= 208, not <= 224. A 222px crop pads to
  // 240 and silently becomes four tiles.
  if (wl.crop) {
    const [ch, cw] = wl.crop;
    const y0 = Math.max(0, (H - ch) >> 1), x0 = Math.max(0, (W - cw) >> 1);
    planes = planes.map((p) => {
      const out = new Float32Array(ch * cw);
      for (let y = 0; y < ch; y++) {
        for (let x = 0; x < cw; x++) out[y * cw + x] = p[(y0 + y) * W + (x0 + x)];
      }
      return out;
    });
    H = ch; W = cw;
  }
  // Which plane carries the channel to segment is a property of the file, not a
  // convention — Composite.tif keeps cytoplasm on plane 1 and nuclei on plane 2, and
  // segmenting plane 0 at the wrong diameter silently returns zero objects rather than
  // failing. The notebook's seed script is the reference for these choices.
  const gray = planes[wl.plane ?? 0];
  const chan2 = wl.chan2Plane != null ? planes[wl.chan2Plane] : null;
  const opts = { diameter: 30, min_size: 15, ...wl.opts, chan2 };

  cp._stages = {};
  const recorder = new DispatchRecorder(cp.device);
  const detach = attach(cp, recorder);

  // Warm-up: shader compilation, buffer-pool fill, GPU clock ramp. Discarded.
  await cp.segmentImage(gray, H, W, opts);
  await recorder.drain();
  recorder.reset();

  const runs = [];
  const overlaps = [];
  for (let i = 0; i < repeats; i++) {
    const res = await cp.segmentImage(gray, H, W, opts);
    const records = await recorder.drain();
    let n = 0;
    for (let k = 0; k < res.labels.length; k++) if (res.labels[k] > n) n = res.labels[k];
    runs.push({ records, stages: { ...res.timings.stages, total: res.timings.total }, objects: n });
    // Read the overlap check *before* resetting — reset() clears the records it is
    // computed from, and taking it at the end silently produced an empty report.
    overlaps.push(recorder.overlapReport());
    recorder.reset();
  }
  detach();
  cp._stages = null;

  // Element-wise median across runs, keyed by position in the deterministic sequence.
  const len = Math.min(...runs.map((r) => r.records.length));
  const dispatches = [];
  for (let i = 0; i < len; i++) {
    const base = runs[0].records[i];
    const nss = runs.map((r) => r.records[i].ns);
    dispatches.push({ ...base, ns: median(nss), nsSpread: spread(nss) });
  }
  const stageKeys = new Set(runs.flatMap((r) => Object.keys(r.stages)));
  const stages = {};
  for (const k of stageKeys) stages[k] = median(runs.map((r) => r.stages[k] || 0));

  const annotated = dispatches.map((d) => annotate(d, { waveCapacity }));
  return {
    workload: { ...wl, W, H, channels: chan2 ? 2 : 1, opts: { ...opts, chan2: !!chan2 } },
    objects: runs[0].objects,
    repeats,
    totalMsSpread: spread(runs.map((r) => r.stages.total)),
    stages,
    dispatches: annotated,
    gpuBusyNs: annotated.reduce((s, r) => s + r.ns, 0),
    summary: summarise(annotated),
    overlap: {
      encoders: overlaps.reduce((s, o) => s + o.encoders, 0),
      medianRatio: median(overlaps.map((o) => o.medianRatio).filter((v) => v != null)),
      maxRatio: Math.max(...overlaps.map((o) => o.maxRatio ?? 0)),
      serial: overlaps.every((o) => o.serial !== false),
    },
  };
}

/**
 * Cross-check on a second engine (task 0.7).
 *
 * StarDist is a different U-shaped network with a *separately written* conv kernel of
 * the same 16x16 / BLK=8 shape. If its dispatches land in the same class at a similar
 * attainment, the Phase 0 finding is about this family of kernels rather than about
 * cyto3 — which is the question worth answering before generalising anything.
 *
 * Deliberately thin: one image, no gates, no Amdahl split. It exists to falsify
 * "cyto3-specific", not to profile StarDist properly.
 */
export async function profileStarDist(device, wl, roofs, floor, { repeats = 3 } = {}) {
  const { StarDistWebGPU } = await import("../stardist.js");
  const sd = await StarDistWebGPU.load(
    new URL("../../weights/stardist-fluo/", import.meta.url).href, { device });
  const { W, H, planes } = await loadPlanes(wl.url);
  const gray = planes[wl.plane ?? 0];
  const opts = { prob_thresh: 0.479, nms_thresh: 0.3, ...wl.opts };

  const recorder = new DispatchRecorder(device);
  const detach = attach(sd, recorder);
  await sd.segmentImage(gray, H, W, opts);
  await recorder.drain();
  recorder.reset();

  const runs = [];
  for (let i = 0; i < repeats; i++) {
    await sd.segmentImage(gray, H, W, opts);
    runs.push(await recorder.drain());
    recorder.reset();
  }
  detach();

  const len = Math.min(...runs.map((r) => r.length));
  const dispatches = [];
  for (let i = 0; i < len; i++) {
    dispatches.push({ ...runs[0][i], ns: median(runs.map((r) => r[i].ns)) });
  }
  const annotated = dispatches.map((d) => annotate(d, { waveCapacity: roofs.waveCapacity }));
  const cls = classify(annotated, roofs, floor);
  return {
    engine: "stardist-fluo", workload: { ...wl, W, H },
    gpuBusyNs: annotated.reduce((s, r) => s + r.ns, 0),
    dispatches: cls.rows,
    classification: {
      buckets: cls.buckets, causes: cls.causes,
      timeWeightedAttainment: cls.timeWeightedAttainment,
      coverage: cls.coverage, recoverableShare: cls.recoverableShare,
    },
  };
}

/** Full Phase 0 run: machine roofs once, then every workload. */
export async function runPhase0({ workloads, repeats = 5, gateThresholds = {}, crossCheck = null } = {}) {
  const device = await sharedDevice();
  if (!device.features.has("timestamp-query")) {
    throw new Error("timestamp-query unavailable — per-dispatch timing is impossible on "
      + "this browser/GPU, and Phase 0 cannot be completed here");
  }

  const roofs = await measureRoofs(device);
  const floor = await launchFloor(device);
  const cp = await CellposeWebGPU.load(undefined, { device });

  const results = [];
  for (const wl of workloads) {
    const r = await profileWorkload(cp, wl, { repeats, waveCapacity: roofs.waveCapacity });
    const cls = classify(r.dispatches, roofs, floor, gateThresholds);
    results.push({
      ...r,
      dispatches: cls.rows,
      classification: {
        buckets: cls.buckets,
        causes: cls.causes,
        timeWeightedAttainment: cls.timeWeightedAttainment,
        coverage: cls.coverage,
        recoverableShare: cls.recoverableShare,
      },
      gates: gates(r.stages, r.gpuBusyNs, cls, gateThresholds),
    });
  }

  const stardist = crossCheck
    ? await profileStarDist(device, crossCheck, roofs, floor)
    : null;

  return {
    generated: new Date().toISOString(),
    gpu: adapterDescription(device),
    userAgent: navigator.userAgent,
    roofs, launchFloor: floor,
    results, stardist,
  };
}
