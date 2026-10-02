// Headless regression driver for the notebook.
//
// Boots notebook.html in Chrome with WebGPU, optionally stages input files into the
// Pyodide filesystem (headless can't use the folder picker), runs every cell, and
// reports each cell's status and output. Exits nonzero if any cell errors.
//
//   node tools/drive.mjs                       # stages tools/test_image.tif, runs the seed
//   node tools/drive.mjs --stage /path/to/other.tif
//   node tools/drive.mjs --url http://localhost:8765/notebook.html --keep
//   node tools/drive.mjs --swiftshader          # no usable GPU: WebGPU on the CPU, very slow
//
// Requires a static server on the repo root and puppeteer-core (installed under
// tools/, since that is where this repo keeps its npm dev tools):
//   python3 -m http.server 8765 &
//   npm --prefix tools install
import * as notebook from "./notebook.mjs";

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
const STAGE_FILES = STAGE.length ? STAGE : ["tools/test_image.tif"];

const CHROME = flag("--chrome", process.env.CHROME_PATH ||
  "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome");

// The nuclear/cytoplasm ratio pipeline used to ship as the notebook's own seeded
// example; the app now starts every fresh notebook empty, so this driver carries its
// own copy as a fixture — injected below only when the page has no cells already —
// to keep the documented test_image.tif baseline (190 cells, 183 nuclei, 173 kept)
// reproducible without a manual click. Written exactly as upstream cellpose/stardist
// docs would have it — no `await` — the kernel's auto-await rewrite makes that
// invisible regardless of whether the browser has JS Promise Integration.
const SEED_PIPELINE = [
`import numpy as np
import pandas as pd
from skimage import io, measure
from cellpose import models

img = io.imread("test_image.tif")     # ← or your own file; see the workspace bar above
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

const browser = await notebook.launch({
  chrome: CHROME, headful: has("--headful"), swiftshader: has("--swiftshader"),
});

let failed = false;
try {
  console.log(`→ ${URL_}`);
  // Start from the example pipeline rather than whatever a previous manual session
  // left in localStorage, so a run is reproducible.
  const { page, logs } = await notebook.openNotebook(browser, URL_, { timeout: TIMEOUT, fresh: !has("--keep") });
  console.log("✓ booted");

  // A fresh notebook (no saved session) starts empty now that the app doesn't ship its
  // own example — seed this driver's fixture pipeline so the run stays reproducible.
  if (await notebook.cellCount(page) === 0) {
    await notebook.appendCells(page, SEED_PIPELINE);
    console.log(`✓ seeded ${SEED_PIPELINE.length} cells (no saved notebook)`);
  }

  for (const b of await notebook.banners(page)) console.log("! " + b);

  for (const rel of STAGE_FILES) {
    const name = rel.split("/").pop();
    await notebook.stageUrl(page, `${BASE}/${rel}`, name);
    console.log(`✓ staged ${name}`);
  }

  console.log("\n→ run all");
  await notebook.runAll(page, { timeout: TIMEOUT });

  const cells = await notebook.readCells(page);

  console.log("");
  for (const c of cells) {
    const mark = c.state === "done" ? "✓" : c.state === "error" ? "✗" : "·";
    const extras = [c.figures.length ? `${c.figures.length} figure(s)` : null, c.table ? "table" : null]
      .filter(Boolean).join(", ");
    console.log(`${mark} cell ${c.i} [${c.state}] ${c.took}${extras ? "  " + extras : ""}`);
    if (c.out) console.log("    " + c.out.slice(0, 600).replace(/\n/g, "\n    "));
    if (c.state === "error") failed = true;
    if (c.state !== "done" && c.state !== "error") failed = true;
  }

  const files = await notebook.producedFiles(page);
  if (files.length) console.log("\nproduced: " + files.map((f) => `⤓ ${f}`).join(", "));

  if (logs.some((l) => l.startsWith("PAGEERROR"))) failed = true;
  if (has("--logs")) {
    console.log("\n--- browser console ---");
    for (const l of logs) console.log("  " + l);
  }
} finally {
  await browser.close();
}

console.log(failed ? "\nFAILED" : "\nOK");
process.exit(failed ? 1 : 0);
