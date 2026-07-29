// Notebook chrome: theme, workspace folder, toolbar, downloads, boot orchestration.
import * as kernel from "../kernel.js";
import * as cells from "./cells.js";

const $ = (id) => document.getElementById(id);

// ---- seed script --------------------------------------------------------------------
// The original nuclear/cytoplasm ratio script, split at its own comment boundaries —
// they were already natural cell breaks. Written exactly as upstream cellpose/stardist
// docs would have it — no `await` — regardless of whether the browser has JS Promise
// Integration; the kernel's auto-await rewrite (see shims/_ptero_autoawait.py) makes
// that invisible.
const SEED = [
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

// ---- theme ---------------------------------------------------------------------------
let isDark = (() => {
  const stored = localStorage.getItem("ptero-theme");
  if (stored) return stored === "dark";
  return window.matchMedia?.("(prefers-color-scheme: dark)").matches ?? true;
})();

function applyTheme(dark) {
  isDark = dark;
  document.body.classList.toggle("theme-light", !dark);
  $("themetoggle").textContent = dark ? "🌙 Dark" : "☀️ Light";
  localStorage.setItem("ptero-theme", dark ? "dark" : "light");
  cells.applyTheme(dark);
}

// ---- workspace + downloads -------------------------------------------------------------
// Files the notebook produced, offered as downloads. Without a mounted folder this is
// the only way results leave the page, so it is not optional chrome.
const PRODUCED = /\.(csv|tif|tiff|png|json|txt|xlsx)$/i;
let knownFiles = new Set();

async function refreshWorkspace() {
  const names = await kernel.syncWorkspace();
  const info = $("workspace-info");
  if (kernel.hasWorkspace()) {
    info.textContent = names.length
      ? `Workspace: ${names.join(", ")}`
      : "Workspace is empty — copy an image into that folder, then hit ↻.";
  } else {
    info.textContent = "No workspace folder. Cells run against an in-memory filesystem; " +
      "anything they write shows up below as a download.";
  }

  const dl = $("downloads");
  dl.replaceChildren();
  for (const name of names) {
    if (!PRODUCED.test(name) || !knownFiles.has(name)) continue;
    const a = document.createElement("a");
    a.className = "download";
    a.textContent = `⤓ ${name}`;
    a.href = URL.createObjectURL(new Blob([kernel.readFile(name)]));
    a.download = name;
    dl.appendChild(a);
  }
}

// Only offer downloads for files a *cell run* produced. Diffing the listing across
// each run (rather than once against boot) means mounting a folder of images, or
// dropping a new file in mid-session, never makes those inputs look like results.
let filesBeforeRun = new Set();

async function beforeRun() {
  filesBeforeRun = new Set(await kernel.syncWorkspace());
}

async function noteNewFiles() {
  for (const n of await kernel.syncWorkspace()) {
    if (!filesBeforeRun.has(n)) knownFiles.add(n);
  }
}

async function chooseWorkspace() {
  if (!window.showDirectoryPicker) {
    setStatus("folder picker needs Chrome or Edge");
    return;
  }
  let handle;
  try {
    handle = await window.showDirectoryPicker({ mode: "readwrite" });
  } catch {
    return;   // user cancelled
  }
  if (await handle.requestPermission({ mode: "readwrite" }) !== "granted") {
    setStatus("read/write access denied");
    return;
  }
  await kernel.mountWorkspace(handle);
  $("choosews").textContent = `📁 ${handle.name}`;
  // A newly mounted folder's existing contents are inputs, not results.
  knownFiles = new Set();
  await refreshWorkspace();
}

// ---- status ------------------------------------------------------------------------------
function setStatus(msg) { $("stat").textContent = msg; }

// ---- boot ---------------------------------------------------------------------------------
export async function start() {
  cells.mount($("cells"), { dark: isDark });
  cells.setRunHooks({
    before: beforeRun,
    after: async () => { await noteNewFiles(); await refreshWorkspace(); },
  });

  $("themetoggle").addEventListener("click", () => applyTheme(!isDark));
  $("choosews").addEventListener("click", chooseWorkspace);
  $("refreshws").addEventListener("click", refreshWorkspace);
  $("addcell").addEventListener("click", () => {
    const c = cells.appendCell("");
    c.view.focus();
  });
  $("runall").addEventListener("click", async () => {
    $("runall").disabled = true;
    try { await cells.runAll(); } finally { $("runall").disabled = false; }
  });
  $("resetcells").addEventListener("click", () => {
    if (!confirm("Replace all cells with the example pipeline? This discards your edits.")) return;
    cells.resetCells(SEED);
  });
  applyTheme(isDark);

  if (!navigator.gpu) {
    setStatus("no WebGPU");
    document.body.prepend(banner(
      "This page needs WebGPU. Use Chrome or Edge (or Safari 18+) — segmentation cells " +
      "will fail without it, though the rest of the Python still runs."));
  }

  setStatus("booting Python…");
  try {
    await kernel.boot({ stdout: (m) => console.log(m) });
  } catch (e) {
    setStatus("boot failed");
    document.body.prepend(banner(`Couldn't start the Python kernel: ${e.message}`));
    throw e;
  }
  // Restore the previous session's cells, or seed the example pipeline.
  const saved = cells.loadSavedCells();
  if (saved) {
    for (const { source, author } of saved) cells.appendCell(source, author);
  } else {
    for (const src of SEED) cells.appendCell(src);
  }

  // Loaded dynamically so a session that never opens the assistant doesn't pay for it.
  const { mountChat } = await import("../agent/chat.js");
  mountChat();

  setStatus("ready");
  await refreshWorkspace();
}

function banner(text, kind = "error") {
  const d = document.createElement("div");
  d.className = `banner banner-${kind}`;
  d.textContent = text;
  return d;
}
