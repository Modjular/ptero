// Phase 1 driver: benchmark conv kernel variants against the shipping kernel.
//
//   python3 -m http.server 8765 &
//   node tools/convbench.mjs                 # every variant, every shape
//   node tools/convbench.mjs --shapes 4      # top 4 shapes only (74% of conv time)
//   node tools/convbench.mjs --only baseline,static_acc
//   node tools/convbench.mjs --per-shape     # full per-shape table
import { writeFileSync, mkdirSync } from "node:fs";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const flag = (n, d = null) => { const i = args.indexOf(n); return i === -1 ? d : args[i + 1]; };
const has = (n) => args.includes(n);

const BASE = flag("--base", "http://localhost:8765");
const NSHAPES = flag("--shapes", null);
const ONLY = flag("--only", null);
const CHROME = flag("--chrome", process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");

const browser = await puppeteer.launch({
  executablePath: CHROME, headless: !has("--headful"),
  args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--no-sandbox"],
});

let out;
try {
  const page = await browser.newPage();
  page.on("pageerror", (e) => console.error("  PAGEERROR", e.message));
  page.on("console", (m) => { if (has("--logs")) console.log("  [page]", m.text()); });
  await page.goto(`${BASE}/index.html`, { waitUntil: "domcontentloaded" });

  out = await page.evaluate(async (nshapes, only) => {
    try {
      const { sharedDevice } = await import("/src/gpu.js");
      const { benchVariants, SHAPES } = await import("/src/profile/convbench.js");
      const { VARIANTS } = await import("/src/profile/conv-variants.js");
      const { intensitySweep } = await import("/src/profile/roofs.js");
      const device = await sharedDevice();

      // The roof, measured here rather than read from the Phase 0 JSON, so attainment is
      // against this session's thermal state rather than a different one's.
      const sweep = await intensitySweep(device);
      const roofGflops = sweep.peakGflops;

      let variants = VARIANTS;
      if (only) {
        const want = only.split(",");
        variants = VARIANTS.filter((v) => want.includes(v.id));
        // The first variant is the correctness reference and the speedup denominator, so
        // it has to be present whatever the filter says.
        if (!variants.some((v) => v.id === VARIANTS[0].id)) variants.unshift(VARIANTS[0]);
      }
      const shapes = nshapes ? SHAPES.slice(0, Number(nshapes)) : SHAPES;

      // Compile everything up front so a syntax error is reported as one clear message
      // rather than surfacing as a device error mid-benchmark. Shape-specialised
      // variants generate a different source per distinct shape, so check each.
      const compileErrors = [];
      for (const v of variants) {
        const sources = typeof v.wgsl === "function"
          ? [...new Set(shapes.filter((s) => !v.applicable || v.applicable(s)).map((s) => v.wgsl(s)))]
          : [v.wgsl];
        for (const code of sources) {
          const info = await device.createShaderModule({ code, label: v.id }).getCompilationInfo();
          for (const m of info.messages) {
            if (m.type === "error") {
              const line = code.split("\n")[m.lineNum - 1] || "";
              compileErrors.push(`${v.id}:${m.lineNum}:${m.linePos}: ${m.message}\n      ${line.trim()}`);
            }
          }
        }
      }
      if (compileErrors.length) return { compileErrors };

      const r = await benchVariants(device, variants, { shapes, roofGflops });
      return { roofGflops, ...r };
    } catch (e) {
      return { error: e.message, stack: e.stack };
    }
  }, NSHAPES, ONLY);
} finally {
  await browser.close();
}

if (out.compileErrors) {
  console.error("shader compilation failed:");
  for (const e of out.compileErrors) console.error("  " + e);
  process.exit(1);
}
if (out.error) {
  console.error("FAILED:", out.error); console.error(out.stack); process.exit(1);
}

const pct = (x) => x == null ? "—" : (x * 100).toFixed(2) + "%";
console.log(`\nroof ${out.roofGflops.toFixed(0)} GFLOP/s (measured this session)\n`);
console.log("variant".padEnd(18) + "ok".padEnd(6) + "GFLOP/s".padStart(9)
  + "attain".padStart(9) + "proj speedup".padStart(14) + "  what");
for (const s of out.summary) {
  if (s.unavailable) {
    console.log(s.variant.padEnd(18) + "n/a".padEnd(6) + "—".padStart(9) + "—".padStart(9)
      + "—".padStart(14) + "  " + s.what);
    console.log("".padEnd(18) + "   " + s.unavailable);
    continue;
  }
  console.log(
    s.variant.padEnd(18)
    + (s.allOk ? "ok" : "FAIL").padEnd(6)
    + s.weightedGflops.toFixed(0).padStart(9)
    + pct(s.weightedAttainment).padStart(9)
    + (s.projectedSpeedup.toFixed(2) + "x").padStart(14)
    + "  " + s.what);
  if (!s.allOk) console.log("".padEnd(18) + "   failing shapes: " + s.failures.join(", "));
}

if (has("--per-shape")) {
  console.log("\nper shape:");
  const shapes = [...new Set(out.rows.map((r) => JSON.stringify(r.shape)))];
  for (const sk of shapes) {
    const s = JSON.parse(sk);
    console.log(`\n  ${s.Cin}->${s.Cout} @ ${s.H}x${s.W} k${s.K}  (${(s.share * 100).toFixed(1)}% of conv time)`);
    for (const r of out.rows.filter((r) => JSON.stringify(r.shape) === sk)) {
      console.log("    " + r.variant.padEnd(18)
        + (r.ns / 1e6).toFixed(3).padStart(9) + " ms"
        + r.gflops.toFixed(0).padStart(8) + " GFLOP/s"
        + pct(r.attainment).padStart(9)
        + (r.speedup.toFixed(2) + "x").padStart(9)
        + (r.ok ? "" : `   FAIL relerr=${r.maxRel.toExponential(1)}`));
    }
  }
}

mkdirSync("results", { recursive: true });
const f = `results/convbench-${new Date().toISOString().slice(0, 10)}.json`;
writeFileSync(f, JSON.stringify(out, null, 2));
console.log(`\nwrote ${f}`);
