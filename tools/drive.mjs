// Headless regression driver for the notebook.
//
// Boots notebook.html in Chrome with WebGPU, optionally stages input files into the
// Pyodide filesystem (headless can't use the folder picker), runs every cell, and
// reports each cell's status and output. Exits nonzero if any cell errors.
//
//   node tools/drive.mjs                       # run the seeded cells as-is
//   node tools/drive.mjs --stage demo/images/Composite.tif
//   node tools/drive.mjs --url http://localhost:8765/notebook.html --keep
//
// Requires a static server on the repo root and puppeteer-core:
//   python3 -m http.server 8765 &
//   npm install
import { spawn } from "node:child_process";
import puppeteer from "puppeteer-core";

const args = process.argv.slice(2);
const flag = (name, fallback = null) => {
  const i = args.indexOf(name);
  return i === -1 ? fallback : args[i + 1];
};
const has = (name) => args.includes(name);

const BASE = flag("--base", "http://localhost:8765");
const URL_ = flag("--url", `${BASE}/notebook.html`);
const TIMEOUT = Number(flag("--timeout", 300)) * 1000;
const STAGE = args.filter((a, i) => args[i - 1] === "--stage");

const CHROME = flag("--chrome", process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");

// The nuclear/cytoplasm ratio pipeline used to ship as the notebook's own seeded
// example; the app now starts every fresh notebook empty, so this driver carries its
// own copy as a fixture — injected below only when the page has no cells already —
// to keep the documented Composite.tif baseline (190 cells, 183 nuclei, 173 kept)
// reproducible without a manual click. Written exactly as upstream cellpose/stardist
// docs would have it — no `await` — the kernel's auto-await rewrite makes that
// invisible regardless of whether the browser has JS Promise Integration.
const SEED_PIPELINE = [
`import numpy as np
import pandas as pd
from skimage import io, measure
from cellpose import models

img = io.imread("Composite.tif")     # ← or your own file; see the workspace bar above
cyto_channel = img[..., 1]
nuc_channel  = img[..., 2]
img.shape`,

`# cyto3 handles both channels — nuclei are found by shrinking the diameter rather than
# by loading a second model (image.sc #114981). This kernel also has StarDist and
# InstanSeg; \`import ptero; ptero.models.list()\` shows what each is good for, and for a
# fluorescent nuclear stain like this one StarDist is the more specialised choice:
#     from stardist.models import StarDist2D
#     nuc_masks, _ = StarDist2D.from_pretrained('2D_versatile_fluo').predict_instances(nuc_channel)
model = models.CellposeModel(gpu=True, model_type='cyto3')
cell_masks, _, _ = model.eval(cyto_channel, diameter=100, channels=[0, 0])
nuc_masks,  _, _ = model.eval(nuc_channel,  diameter=50,  channels=[0, 0])
f"{cell_masks.max()} cells, {nuc_masks.max()} nuclei"`,

`# --- filter cells without nuclei ---
cells_with_nuclei = np.unique(cell_masks[nuc_masks > 0])
filtered_cell_masks = np.where(np.isin(cell_masks, cells_with_nuclei), cell_masks, 0)

# --- cytoplasm = cell minus nucleus ---
cyto_masks = filtered_cell_masks.copy()
cyto_masks[nuc_masks > 0] = 0
f"{len(cells_with_nuclei)} cells kept"`,

`# --- map nucleus id -> cell id ---
mapping_props = measure.regionprops_table(nuc_masks, intensity_image=filtered_cell_masks,
                                          properties=['label', 'intensity_max'])
df_map = pd.DataFrame(mapping_props).rename(columns={'label': 'Nuc_ID', 'intensity_max': 'Cell_ID'})

# --- intensities ---
nuc_props = measure.regionprops_table(nuc_masks, intensity_image=cyto_channel,
                                      properties=['label', 'intensity_mean'])
df_nuc = pd.DataFrame(nuc_props).rename(columns={'label': 'Nuc_ID', 'intensity_mean': 'Nuc_Mean'})

cyto_props = measure.regionprops_table(cyto_masks, intensity_image=cyto_channel,
                                       properties=['label', 'intensity_mean'])
df_cyto = pd.DataFrame(cyto_props).rename(columns={'label': 'Cell_ID', 'intensity_mean': 'Cyto_Mean'})
len(df_map)`,

`df_final = df_map.merge(df_nuc, on='Nuc_ID').merge(df_cyto, on='Cell_ID')
df_final['Ratio_Nuc_Cyto'] = np.where(df_final['Cyto_Mean'] == 0, 0,
                                      df_final['Nuc_Mean'] / df_final['Cyto_Mean'])

df_final.to_csv("Cell_Measurements.csv", index=False)
io.imsave("Filtered_Cell_Labels.tif", filtered_cell_masks.astype(np.uint16))
df_final`,

`import matplotlib.pyplot as plt
from skimage.color import label2rgb

fig, axes = plt.subplots(1, 2, figsize=(10, 5))
axes[0].imshow(label2rgb(filtered_cell_masks, bg_label=0))
axes[0].set_title(f"{len(cells_with_nuclei)} filtered cells")
axes[0].axis('off')
axes[1].hist(df_final['Ratio_Nuc_Cyto'], bins=30)
axes[1].set_title('Ratio_Nuc_Cyto')
plt.show()`,
];

const browser = await puppeteer.launch({
  executablePath: CHROME,
  headless: !has("--headful"),
  // ANGLE-on-Metal is what makes WebGPU work in headless Chrome on macOS; without
  // these the page boots but navigator.gpu is missing and every model call fails.
  args: ["--enable-unsafe-webgpu", "--use-angle=metal", "--no-sandbox"],
});

let failed = false;
try {
  const page = await browser.newPage();
  const logs = [];
  page.on("console", (m) => logs.push(m.text()));
  page.on("pageerror", (e) => { logs.push("PAGEERROR " + e.message); failed = true; });

  console.log(`→ ${URL_}`);
  await page.goto(URL_, { waitUntil: "domcontentloaded" });

  // Start from the example pipeline rather than whatever a previous manual session
  // left in localStorage, so a run is reproducible.
  if (!has("--keep")) {
    await page.evaluate(() => localStorage.removeItem("ptero-cells"));
    await page.reload({ waitUntil: "domcontentloaded" });
  }

  await page.waitForFunction(
    () => document.getElementById("stat")?.textContent === "ready", { timeout: TIMEOUT });
  console.log("✓ booted");

  // A fresh notebook (no saved session) starts empty now that the app doesn't ship its
  // own example — seed this driver's fixture pipeline so the run stays reproducible.
  const hadCells = await page.evaluate(() => document.querySelectorAll(".cell").length > 0);
  if (!hadCells) {
    await page.evaluate(async (seed) => {
      const cellsMod = await import("./src/notebook/cells.js");
      for (const src of seed) cellsMod.appendCell(src);
    }, SEED_PIPELINE);
    console.log(`✓ seeded ${SEED_PIPELINE.length} cells (no saved notebook)`);
  }

  const banners = await page.evaluate(() =>
    [...document.querySelectorAll(".banner")].map((b) => b.textContent));
  for (const b of banners) console.log("! " + b);

  // Stage input files into the Pyodide FS — the folder picker needs a real user
  // gesture, so a headless run has no other way to supply an image.
  for (const rel of STAGE) {
    const name = rel.split("/").pop();
    await page.evaluate(async (url, name) => {
      const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
      window.pyodide.FS.writeFile(name, buf);
    }, `${BASE}/${rel}`, name);
    console.log(`✓ staged ${name}`);
  }

  console.log("\n→ run all");
  await page.click("#runall");
  await page.waitForFunction(
    () => !document.getElementById("runall").disabled, { timeout: TIMEOUT });

  const cells = await page.evaluate(() =>
    [...document.querySelectorAll(".cell")].map((el, i) => ({
      i,
      state: el.dataset.state,
      author: el.dataset.author,
      took: el.querySelector(".cellstat")?.textContent || "",
      out: (el.querySelector(".cellout")?.textContent || "").trim().slice(0, 600),
      figs: el.querySelectorAll(".cellout-fig").length,
      table: !!el.querySelector(".cellout-html table"),
    })));

  console.log("");
  for (const c of cells) {
    const mark = c.state === "done" ? "✓" : c.state === "error" ? "✗" : "·";
    const extras = [c.figs ? `${c.figs} figure(s)` : null, c.table ? "table" : null]
      .filter(Boolean).join(", ");
    console.log(`${mark} cell ${c.i} [${c.state}] ${c.took}${extras ? "  " + extras : ""}`);
    if (c.out) console.log("    " + c.out.replace(/\n/g, "\n    "));
    if (c.state === "error") failed = true;
    if (c.state !== "done" && c.state !== "error") failed = true;
  }

  const files = await page.evaluate(() =>
    [...document.querySelectorAll("#downloads .download")].map((a) => a.textContent));
  if (files.length) console.log("\nproduced: " + files.join(", "));

  if (has("--logs")) {
    console.log("\n--- browser console ---");
    for (const l of logs) console.log("  " + l);
  }
} finally {
  await browser.close();
}

console.log(failed ? "\nFAILED" : "\nOK");
process.exit(failed ? 1 : 0);
