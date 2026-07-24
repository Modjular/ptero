# `pipeline.html` — handoff notes

Written 2026-07-22 to prep `pipeline.html` for extraction into its own repo. This
captures the journey and the *why* behind decisions, not just the *what* — read this
before changing architecture, not just before changing code.

## The goal, in one paragraph

Run real scientific-analysis Python (numpy/pandas/scikit-image today, arbitrary
libraries eventually) **entirely in the browser, entirely locally, no server**. The
driving example is a nuclear/cytoplasm intensity-ratio measurement script that
segments cells with Cellpose, filters by nucleus presence, and computes per-cell
intensity ratios via `regionprops_table` + `pandas`. Python-and-portability is the
actual point (stated explicitly by the user) — this is *not* a "port the logic to
JS" project, it's "make Python's scientific stack run natively client-side, with a
GPU escape hatch for the one thing that needs one (CNN inference)."

## Architecture: Pyodide science kernel + WebGPU inference backend

- **Pyodide runs the user's Python almost verbatim.** `numpy`, `pandas`,
  `scikit-image`, `matplotlib` load as real Pyodide packages; `tifffile` installs at
  runtime via `micropip` (pure-Python, not in the Pyodide lockfile). This means
  `regionprops_table`, `pandas.merge`, `io.imsave`, `matplotlib` plotting — all of it
  runs completely unmodified.
- **The one intercept:** a fake `cellpose` module is injected into `sys.modules`
  before any user code runs (see `SHIM` in `pipeline.html`). Its
  `Cellpose(...).eval(img, diameter, channels)` is `async`, marshals the image to JS,
  and calls the real WebGPU engine.
- **The WebGPU engine is untouched, borrowed as-is:** `cellpose_core.js`, exporting
  `class CellposeWebGPU` — hand-written WGSL for the CNN forward pass, JS for
  flow-dynamics post-processing. Public API:
  ```js
  const cp = await CellposeWebGPU.create();
  cp.loadWeights(manifestJson, weightsArrayBuffer);
  const { labels, output, dP, cellprob, H, W, timings } =
    await cp.segmentImage(grayFloat32, H, W, { diameter, chan2 });
  // labels: Int32Array[H*W], 0 = background. Defaults inside computeMasksGPU:
  // cellprob_threshold=0.0, niter=200, min_size=15, rpad=20, flow_threshold=0.4
  ```
  `segmentImage` already does cellpose's own rescale-by-`30/diameter` → forward →
  resize-flows-back → dynamics, so both cyto (`diameter=100`) and nuclei
  (`diameter=50`, same model) calls work with **zero engine changes**.
- **Model decision:** `cyto3` for *both* the cytoplasm and nuclei channels (nuclei
  detected by swapping which channel goes in + shrinking diameter — a real technique,
  see image.sc thread 114981), not a second model. So no InstanSeg/StarDist port was
  needed for this pipeline (those exist as sibling teaching ports in the parent repo,
  unrelated to this one).

### Files this spinoff actually needs from the parent repo

Only these — everything else in the parent directory (StarDist/InstanSeg ports,
ONNX export, checkpoints, mini_* teaching scripts, ipynb walkthroughs, perf reports)
is unrelated research from the same repo and should **not** come along:

| File | Role |
|---|---|
| `pipeline.html` | Everything: UI, cell shell, Pyodide boot, the shim, the bridge. |
| `style.css` | Theme variables (light/dark) + all component styles. |
| `cellpose_core.js` | `CellposeWebGPU` — WGSL forward pass + JS dynamics. Unmodified. |
| `cyto3_manifest.json` + `cyto3_weights.bin` | Exported, BatchNorm-folded cyto3 weights (~26 MB), fetched at boot. |
| `images/Composite.tif` | Demo image (960×1280, C0=AF/C1=cyto/C2=nuclei) — the default input if the user doesn't pick a file. |

## The Pyodide↔WebGPU bridge — interop gotchas (hard-won, don't re-derive these)

- **Python↔JS boundary is fundamentally async** (WebGPU calls are async), which is
  the *only* deviation from verbatim in the user's script: `await model.eval(...)`
  instead of bare `model.eval(...)`. Removing this `await` requirement (so scripts
  run 100% verbatim) needs a Worker+Atomics bridge or JSPI `run_sync` — **deferred,
  not started**. See "Explicitly deferred" below.
- **Never dunder-name a JS global called from inside a Pyodide class method.**
  `js.__cpSegment` referenced inside a Python class method gets name-mangled by
  Python to `_ClassName__cpSegment` → `AttributeError`. The JS bridge global is
  named `cpSegmentBridge` (no leading dunder) specifically because of this.
- **All cells/scripts share ONE Pyodide global namespace** (`pyodide.runPythonAsync`
  uses `pyodide.globals` by default) — this is a *feature* (it's what makes
  cross-cell variable persistence work with zero extra plumbing) but it's also a
  footgun for our own infrastructure code: the matplotlib figure-capture helper
  originally did `import base64, io` at module scope, and a cell doing
  `from skimage import io` silently clobbered the stdlib `io` name out from under it
  (`AttributeError: module 'skimage.io' has no attribute 'BytesIO'`). Fixed by
  moving those imports *inside* the helper function. **Any future kernel-level
  helper must scope its imports locally, never at the shared global scope.**
- **Marshaling shapes that work:** numpy(f32) proxy → JS via
  `imgProxy.getBuffer('f32').data.slice()` (copy, then `.release()` the wasm-heap
  view). JS `Int32Array` → numpy via `await`, which Pyodide auto-converts to a
  memoryview: `np.asarray(mv, dtype=np.int32).reshape(H, W)`.
- **Rich cell output:** a Python object with `_repr_html_` (e.g. any
  `pandas.DataFrame`) renders as a real HTML `<table>` — no CSV round-trip needed,
  just call `result._repr_html_()` from JS on the PyProxy if it exists (wrap in
  try/catch; `undefined` access on a PyProxy for a missing attribute is safe). Open
  matplotlib figures are popped after each cell run the same way Jupyter's inline
  backend does: `plt.get_fignums()` → `savefig` to a `BytesIO` PNG → base64 → `<img>`.
  Backend is forced to `Agg` once at boot (`matplotlib.use('Agg')`, before any cell
  can `import matplotlib.pyplot`), since Pyodide's interactive canvas backend isn't
  what we want for static figure capture. `plt.show()` is also monkey-patched to a
  no-op at boot (mutating the shared `matplotlib.pyplot` module object's `.show`
  attribute, not a name binding, so it stays patched no matter what a cell imports
  it as) — under plain `Agg`, `plt.show()` throws a `UserWarning:
  FigureCanvasAgg is non-interactive` that looks like an error to a non-technical
  user, for no functional reason (figures are captured regardless of whether
  `plt.show()` was ever called). Real Jupyter's inline backend makes the same call.

## Fidelity status — a known, currently-open gap

Verified end-to-end on `images/Composite.tif` via headless Chrome
(`--enable-unsafe-webgpu --use-angle=metal`) against a real PyTorch `cyto3` desktop
reference: **AP@0.5 = 0.914** (TP=160, FP=3, FN=12) on the filtered cell label maps,
nuc/cyto ratio **Pearson r = 0.85, mean|Δ| = 0.07**. Browser got 176 cells/178
nuclei/173 mapped vs desktop's 190/183/177.

This gap was diagnosed as **living in the WebGPU cyto3 engine on this
out-of-distribution brightfield-ish image** (the same engine scores ~1.0 AP@0.5 on
fluorescent reference data), *not* in the Pyodide/interop layer, which reproduces the
numpy/pandas/skimage math exactly. Leading hypotheses, untested: `normalize99`
percentile-normalization parity, resize/interpolation parity (engine's
`resizeBilinear` vs cellpose's `cv2.resize`), and possibly `models.Cellpose` (has a
`SizeModel`) vs `models.CellposeModel` (no size model, what the desktop reference
script actually used) semantic differences.

**Status: explicitly paused, not abandoned.** The user chose to pivot to the
notebook-cell-shell UX work before closing this gap ("let's close the gap before we
move code into a worker" was the instruction that *preceded* this pivot — the pivot
itself was "how do we make this feel like a notebook," which turned out to be a
bigger and more interesting question in the moment). **Whoever picks this up next
should treat the fidelity gap as unfinished business**, not something that was
solved or deprioritized on purpose beyond "the notebook shell was more urgent."

Reference artifacts from this investigation (present in the parent repo, not yet
copied into any new repo): `desktop_ref.py` (real PyTorch reference),
`compare_fidelity.py` (AP@0.5 + ratio-agreement scorer), `drive_pipeline.mjs`
(**now broken**, see below), `pipeline_compare.png` (3-panel visual comparison).

## The notebook-cell-shell pivot

### Why not adopt marimo or JupyterLite

Both were seriously considered — both are real, offline-capable, Pyodide-in-a-worker
notebook systems. Rejected for now because **the thing worth owning is the cell data
model an agent will eventually act on**, not the notebook chrome. See "The bigger
vision" below for why that matters. Building on marimo/JupyterLite would mean
reverse-engineering their internal cell/execution APIs (built for a human clicking
buttons, not a program driving them) to get an agent the same power a human has.
Owning a plain `{id, source, output}` array means "agent edits cell 3 and reruns it"
is just a function call into state we already control.

### Why CodeMirror 6, not Monaco

CodeMirror is a text-editing library; Monaco is an IDE platform (LSP client
machinery, its own worker protocol, much bigger bundle). Neither marimo nor
JupyterLite use Monaco either — both use CodeMirror 6. For "a box that edits Python
and runs it," CodeMirror gets the editing ergonomics without the IDE weight.

**Not yet vendored for offline use** — currently loaded from `esm.sh` CDN
(`codemirror@6.0.1`, `@codemirror/lang-python@6.1.6`, `@codemirror/theme-one-dark@6.1.2`,
plus `@codemirror/state` for `Compartment`), same pattern as Pyodide's existing CDN
load. This is a known gap against the project's own local-first goal, explicitly
deferred as "fine for a feel-test, revisit once the interaction is validated."

**esm.sh gotcha, already hit once, will bite again if more CodeMirror-family
packages get added:** never pin a shared peer dependency (e.g. `@codemirror/state`)
to an independently-chosen exact version. `codemirror@6.0.1` internally depends on
`@codemirror/state@^6.0.0`; importing `@codemirror/state@6.4.1` separately created a
*second*, incompatible module instance, and CodeMirror's extension system does
identity checks — result was a cryptic `Unrecognized extension value in extension
set` runtime error. Fix: import shared peers using the exact same semver range the
consuming package uses (`@codemirror/state@^6.0.0`), so esm.sh resolves both to the
identical concrete build. Check what range a package expects via
`curl -s https://esm.sh/<package>@<version> | head` (the stub shows its raw import
specifiers) before picking a version for anything that shares CodeMirror internals.

### Theming: shared CSS variables + a matching CodeMirror theme

`style.css` defines a semantic variable set (`--bg-base`, `--accent-primary`,
`--border-subtle`, etc.), dark by default, with a `body.theme-light` class
overriding them for light mode. A 🌙/☀️ button toggles that class, persists the
choice to `localStorage` (`pipeline-theme`), and initializes from
`prefers-color-scheme` on first visit. CodeMirror doesn't know about CSS variables,
so it's kept in sync separately: a single `Compartment` holds the CM theme
extension (`oneDark` or none) per editor, and toggling calls
`cell.view.dispatch({ effects: themeCompartment.reconfigure(...) })` on every
currently-mounted cell.

### Cells persist across reloads (localStorage, source only)

Cell *source* (not output — figures/tables are cheap to regenerate by re-running,
and would blow through localStorage's ~5-10MB quota fast) is saved to
`localStorage` under `pipeline-cells`: debounced 500ms after each keystroke (via
CodeMirror's `EditorView.updateListener`), immediately on add/delete. On boot,
`seedCells()` restores from `localStorage` if present, falling back to the
hardcoded `SEED_CELLS` script otherwise. **Deliberately no "reset to default
cells" escape hatch** — if that turns out to be needed, it wasn't an oversight,
it was explicitly deprioritized. Stated future direction (not started): a real
Ctrl-S save to a user-picked file (via `showSaveFilePicker`/the same File System
Access API the workspace folder already uses), keeping the file handle around for
autosave from then on — more local-first than a browser-profile-scoped
`localStorage` blob, and worth doing once the workspace-folder pattern is trusted.

### The cell model as built

Seed cells are the original script split at its own existing comment boundaries
(they were already natural cell breaks) — 5 pipeline cells (load → segment →
filter/cytoplasm → mapping/intensities → ratio+export) plus a 6th added later for
visualization (`skimage.color.label2rgb` overlay + a ratio histogram via
matplotlib). Each cell: `{ id, author: 'user', view (CodeMirror EditorView), wrapEl,
statusEl, outEl }`. `author` is populated but **not yet used anywhere in the UI** —
kept because it's a one-line addition now vs. a retrofit later (see below).

Execution is fully serial against one persistent `pyodide` instance — no
reactivity, no dependency graph. `▶` runs one cell; `▶▶ Run all` runs top-to-bottom,
stopping at the first error. stdout/stderr are captured per-cell via a
`activeCellBuffer` array that the shared stdout handler pushes into while set.
Output panels (the old hardcoded "Filtered cell labels" canvas and "Measurements"
table) were **removed entirely** in favor of cells producing their own rich output
(see interop gotchas above) — the only thing still driven by a post-run check rather
than a cell's own output is the CSV/TIF download links, which reappear whenever
those files exist on the Pyodide FS, regardless of which cell wrote them.

**Explicitly out of scope so far:** reactivity/auto-rerun, `.ipynb` import/export,
cell reordering/drag, widgets, agent read/write hooks, offline-vendored
CodeMirror/Pyodide, multi-kernel.

## The bigger vision (why the architecture choices above matter)

Stated goal, paraphrased: an **agent-supervised, iterative segmentation workflow** —
closer to a music producer partnering with an artist than a one-shot script. The
target user is a lab tech with intent but not expertise, who wants to "point vaguely"
at a region ("is this real signal or an artifact?") and get an agent that walks them
through cell-counting with regular feedback, not a one-shot answer.

Key design conclusion from that discussion (not yet built, but shapes what's worth
protecting now): **don't screenshot the whole notebook and ask a vision model to
guess what was clicked.** The segmentation output is already a label map — a click
resolves to an exact `Cell_ID` via direct array lookup (`cell_masks[y, x]`), free,
deterministic, no vision model needed. Vision earns its keep only for the genuinely
perceptual judgment call itself ("is this real"), where a **targeted crop** from the
raw intensity channels (not a full noisy screenshot) with the mask boundary overlaid
is the right unit to hand to an agent — small, cheap, precise. Structured data
(regionprops, ratios, neighbor distances) stays structured; only the actual
perceptual question goes to an image.

This likely means the "notebook" (code cells) and the "point-and-judge annotation
surface" (closer to CVAT/napari) end up as two UI surfaces sharing one underlying
data model (label maps + dataframe), not one widget. Not resolved, not urgent — but
it's *why* the cell `author` field and "keep label maps addressable by ID all the
way to the renderer" were flagged as cheap insurance worth keeping now.

## Known broken / known gaps (be aware before touching)

- **`drive_pipeline.mjs`** (the headless puppeteer regression driver) targets a
  single `#run` button that no longer exists after the cell-shell rewrite. Broken,
  not updated — deliberately, since headless re-verification wasn't the goal of that
  pass (hands-on feel was). Needs a rewrite (probably: boot-wait, click `#runall`,
  wait for all `.cellstat` to read `done`/`error`) before it's useful again.
- **Local-first isn't fully true yet**: Pyodide (`pyodide.js`, full package set) and
  CodeMirror both load from CDN (`jsdelivr`, `esm.sh`). The WebGPU engine and weights
  are already fully local. Vendoring the rest is a known, deferred gap.
- Ad-hoc verification during this work used one-off Node/puppeteer smoke-test
  scripts written to the session scratchpad and cleaned up after — not checked in,
  not a substitute for a real driver.

## The ~80MB-before-you've-run-anything problem

Real, worth fixing, not yet started. `boot()` currently does two things eagerly,
unconditionally, before the user has run a single cell:
- `pyodide.loadPackage(['numpy', 'pandas', 'scikit-image', 'matplotlib', ...])` —
  every package the *seed* cells might need, loaded upfront regardless of what
  the user's actual cells do.
- `fetch('cyto3_weights.bin')` — the 26 MB cyto3 weights, even if the user's
  session never calls `cellpose.eval()` at all (a real possibility once this
  becomes a general "run arbitrary scientific Python" tool rather than just this
  one pipeline — see "The bigger vision" above).

Python doesn't tree-shake, so there's no free lunch on trimming what's *inside*
numpy/pandas/scikit-image — but loading everything eagerly, upfront, was a
choice made for a first spike, not a hard constraint. Two concrete, genuinely
small fixes:
- **Pyodide has `pyodide.loadPackagesFromImports(code)`** — scans a code string
  for `import` statements and fetches only the packages that code actually
  needs; already-loaded packages are a no-op on repeat calls. Swap the eager
  `loadPackage([...])` call in `boot()` for calling `loadPackagesFromImports(src)`
  right before each cell's `runPythonAsync(src)` in `runCell()`. First boot then
  becomes just the Pyodide core + the shim; `numpy`/`pandas`/`matplotlib`/etc.
  download lazily, spread across the session, the first time a cell actually
  imports them.
- **Lazy-init the WebGPU engine + weights fetch.** Move `CellposeWebGPU.create()`
  and the `cyto3_weights.bin` fetch out of `boot()` and into the `cellpose`
  shim's `eval()`, triggered on its *first real call*, not on page load.
  Sessions that never touch cellpose never pay the 26 MB.

Both are a handful of lines moved, not new architecture — worth doing before
handing this to anyone who isn't on a fast connection.

## Suggested next steps for whoever picks this up

Bias picks here toward **high-impact, low-effort** — this project's whole ethos
so far (CodeMirror over Monaco, a real folder over an in-app file browser, no
marimo/JupyterLite) has been "don't add chrome the interaction doesn't need."
Keep that discipline: none of the below should meaningfully double the codebase.

1. **Cell reordering.** Add/delete exist; reordering doesn't. Simplest version:
   up/down arrows per cell (array `splice` + re-append to `#cells` in the new
   order) — full drag-and-drop is a bigger lift (dragover/dragend wiring, drop
   indicators) and not obviously worth it yet for a handful of cells.
2. **Strip the per-cell header bar, go marimo-style.** Replace the persistent
   `.cellbar` (▶ / status tag / ×) with a ▶ and × that only appear on hover,
   docked to the cell's right edge, out of the way otherwise. Mostly CSS
   (`opacity: 0` + `:hover`/`:focus-within` reveal) plus relocating the two
   buttons out of the bar div — not a rewrite. The `.cellstat` status tag needs
   a new home if the bar goes away (e.g. a thin colored left-border on the cell:
   idle/running/done/error as a border color, no persistent text).
3. **Per-cell compute timer** (nice-to-have, do if time allows). While a cell
   is `running…`, tick a live elapsed-time display (`setInterval` updating a
   text node, cleared on completion) — the classic "compute is happening"
   notebook affordance. Small and self-contained; skip it if it doesn't fit.
4. **Remove the on-screen `#log` panel entirely, and generally tighten
   everything above the cells.** The log is developer-facing only — anyone
   debugging this has the real DevTools console, and everything it shows
   already also goes to `console.log` via the existing `log()` helper.
   Keeping a terminal-like panel on-screen works against the explicit goal of
   this UI reading as *a classic notebook*, not a dev console. This one's
   decided, not a maybe: delete `#log` and the DOM-append half of `log()`,
   keep the `console.log` half. Treat it as the concrete first cut of a
   broader "the top of the page is for the builder, not the user" decluttering
   pass — exact scope of the rest TBD.
5. Fix the eager package/weights loading described above — genuinely
   high-impact, low-effort, and undercuts the local-first pitch as long as it
   stays unfixed.
6. Decide whether to close the AP@0.5 = 0.914 fidelity gap before or after
   further UI work — it was paused, not resolved, and the hypotheses in that
   section are untested.
7. If starting on the agent-supervision vision: the click→`Cell_ID` lookup and
   targeted-crop-not-screenshot approach above is the design conclusion to
   build from, not re-litigate.
8. Fix or replace `drive_pipeline.mjs` before relying on it again — it targets
   a `#run` button that no longer exists.
9. If starting the new repo: bring over exactly the five files listed near the
   top, plus this doc. Don't bring the parent repo's other model
   ports/experiments.
