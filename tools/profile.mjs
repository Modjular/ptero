// Phase 0 measurement driver.
//
// Boots Chrome with WebGPU, runs src/profile/run.js against a fixed benchmark set, and
// writes the results as JSON for profile.html to render. All the measurement logic lives
// in src/profile/ — this file only launches the browser and saves the output.
//
//   python3 -m http.server 8765 &
//   node tools/profile.mjs                       # the frozen benchmark set
//   node tools/profile.mjs --repeats 9 --headful
//   node tools/profile.mjs --only composite
//
// The workloads are frozen here on purpose (task 0.1): one image on the single-tile fast
// path and one that forces the multi-tile path, because the per-tile pipeline drain only
// exists in the latter and is one of the things Phase 0 is trying to price.
import { writeFileSync, mkdirSync } from "node:fs";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const flag = (n, d = null) => { const i = args.indexOf(n); return i === -1 ? d : args[i + 1]; };
const has = (n) => args.includes(n);

const BASE = flag("--base", "http://localhost:8765");
const REPEATS = Number(flag("--repeats", 5));
const ONLY = flag("--only", null);
const OUT = flag("--out", null);
const CHROME = flag("--chrome", process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");

// The frozen benchmark set. `diameter` is recorded with the results because it sets the
// working resolution and therefore the tile count.
const WORKLOADS = [
  {
    id: "single_tile",
    url: `${BASE}/demo/images/cellpose_image_020.png`,
    crop: [208, 208],
    opts: { diameter: 30 },
    why: "centred crop that pads to exactly 224: the ny==nx==1 fast path, no taper-blend",
  },
  {
    // Plane and diameter match the notebook's seed script (src/notebook/ui.js): the
    // cytoplasm channel is plane 1 and the cells are ~100px, which rescales the working
    // resolution down by 30/100 and so uses far fewer tiles than its 1280x960 suggests.
    id: "composite",
    url: `${BASE}/demo/images/Composite.tif`,
    plane: 1,
    opts: { diameter: 100 },
    why: "the notebook's regression case (190 cells), large-diameter rescale path",
  },
  {
    id: "cellpose_020",
    url: `${BASE}/demo/images/cellpose_image_020.png`,
    opts: { diameter: 30 },
    why: "largest frame, 30 tiles with taper blending",
  },
  {
    id: "cellpose_020_diam15",
    url: `${BASE}/demo/images/cellpose_image_020.png`,
    crop: [440, 440],
    opts: { diameter: 15 },
    why: "half the diameter doubles the working resolution — tile count at fixed input size",
  },
];

const selected = ONLY ? WORKLOADS.filter((w) => w.id === ONLY) : WORKLOADS;
if (!selected.length) {
  console.error(`no workload matching --only ${ONLY}; known: ${WORKLOADS.map((w) => w.id).join(", ")}`);
  process.exit(1);
}

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: !has("--headful"),
  // Same flags as tools/drive.mjs — without these, headless Chrome boots the page but
  // navigator.gpu is missing.
  args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--no-sandbox"],
});

let report;
try {
  const page = await browser.newPage();
  page.on("console", (m) => { if (has("--logs")) console.log("  [page]", m.text()); });
  page.on("pageerror", (e) => console.error("  PAGEERROR", e.message));

  await page.goto(`${BASE}/index.html`, { waitUntil: "domcontentloaded" });
  console.log(`→ profiling ${selected.length} workload(s), ${REPEATS} repeats each`);

  // Task 0.7: the same harness against StarDist, to test whether the finding is about
  // cyto3 or about this family of hand-written kernels. Skipped with --no-crosscheck.
  const crossCheck = has("--no-crosscheck") ? null : {
    id: "stardist_fluo", url: `${BASE}/demo/images/Composite.tif`, plane: 2,
    why: "nuclear channel of the regression image, through a separately written conv",
  };

  report = await page.evaluate(async (workloads, repeats, crossCheck) => {
    const { runPhase0 } = await import("/src/profile/run.js");
    try {
      return await runPhase0({ workloads, repeats, crossCheck });
    } catch (e) {
      return { error: e.message, stack: e.stack };
    }
  }, selected, REPEATS, crossCheck);
} finally {
  await browser.close();
}

if (report.error) {
  console.error("\nFAILED:", report.error);
  console.error(report.stack);
  process.exit(1);
}

const stamp = new Date().toISOString().slice(0, 10);
const out = OUT || `results/phase0-${stamp}.json`;
mkdirSync("results", { recursive: true });
writeFileSync(out, JSON.stringify(report, null, 2));

// ---- console summary ----
const pct = (x) => (x * 100).toFixed(1) + "%";
const r = report.roofs;
console.log(`\nGPU: ${report.gpu}`);
console.log(`  compute roof   ${r.sweep.peakGflops.toFixed(0)} GFLOP/s (swept plateau)`);
console.log(`  bandwidth roof ${r.bandwidth.gbps.toFixed(0)} GB/s`);
console.log(`  ridge          ${r.ridgeFromSweep} FLOP/byte (swept), ${r.ridgeFromRatio.toFixed(0)} (ratio)`);
console.log(`  wave capacity  ${r.waveCapacity} workgroups — ${r.wave.note}`);
console.log(`  launch floor   ${report.launchFloor.medianNs} ns`);
if (!r.check.ok) console.log(`  !! roof check FAILED (${r.check.computeVsSweepPlateau.toFixed(2)}) — ${r.check.note}`);

for (const res of report.results) {
  const g = res.gates, c = res.classification;
  console.log(`\n─── ${res.workload.id} — ${res.workload.W}×${res.workload.H}, `
    + `${res.stages.tiles ?? "?"} tile(s), ${res.objects} objects, ±${pct(res.totalMsSpread)} spread`);
  console.log(`  wall ${g.wallMs.toFixed(0)} ms = GPU ${g.gpuBusyMs.toFixed(0)} (${pct(g.gpuShare)})`
    + `  CPU ${g.cpuMs.toFixed(0)} (${pct(g.cpuShare)})  stall ${g.stallMs.toFixed(0)} (${pct(g.stallShare)})`);
  const st = Object.entries(res.stages)
    .filter(([k]) => k !== "total" && k !== "tiles")
    .sort((a, b) => b[1] - a[1]);
  for (const [k, v] of st) console.log(`     ${k.padEnd(14)} ${v.toFixed(1).padStart(8)} ms  ${pct(v / g.wallMs)}`);

  console.log(`  per-tile stall ${g.tileStallMs.toFixed(0)} ms over ${g.tiles ?? "?"} tiles`);
  if (res.overlap.medianRatio != null) {
    console.log(`  pass overlap   sum/span median ${res.overlap.medianRatio.toFixed(2)}, `
      + `max ${res.overlap.maxRatio.toFixed(2)} — ${res.overlap.serial ? "serial, per-pass numbers valid" : "OVERLAPPING, per-pass attribution unsound"}`);
  }
  console.log(`  time-weighted attainment ${c.timeWeightedAttainment == null ? "n/a" : pct(c.timeWeightedAttainment)}`
    + ` (covering ${pct(c.coverage)} of GPU time)`);
  for (const [k, v] of Object.entries(c.buckets)) {
    if (!v.dispatches) continue;
    console.log(`     ${k.padEnd(26)} ${pct(v.share).padStart(6)}  ${v.dispatches} dispatches`);
  }
  console.log(`  below-roof time by cause:`);
  for (const [k, v] of Object.entries(c.causes)) {
    if (!v.ns) continue;
    console.log(`     ${k.padEnd(26)} ${pct(v.share).padStart(6)}`);
  }
  console.log(`  GATE A (Amdahl,  >=${pct(g.gateA.threshold)}): ${g.gateA.pass ? "PASS" : "FAIL"} at ${pct(g.gateA.value)} — ${g.gateA.verdict}`);
  console.log(`  GATE B (headroom,>=${pct(g.gateB.threshold)}): ${g.gateB.pass ? "PASS" : "FAIL"} at ${pct(g.gateB.value)} — ${g.gateB.verdict}`);
}

if (report.stardist) {
  const s = report.stardist, c = s.classification;
  console.log(`\n─── cross-check: ${s.engine} — ${s.workload.W}×${s.workload.H}`);
  console.log(`  GPU busy ${(s.gpuBusyNs / 1e6).toFixed(0)} ms over ${s.dispatches.length} dispatches`);
  console.log(`  time-weighted attainment ${pct(c.timeWeightedAttainment)} (covering ${pct(c.coverage)})`);
  for (const [k, v] of Object.entries(c.buckets)) {
    if (!v.dispatches) continue;
    console.log(`     ${k.padEnd(26)} ${pct(v.share).padStart(6)}  ${v.dispatches} dispatches`);
  }
}

console.log(`\nwrote ${out}`);
