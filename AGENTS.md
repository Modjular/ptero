# AGENTS.md

Guidance for AI coding agents working in this repository. (Formerly `CLAUDE.md`; the
architecture doc that used to live in `docs/ARCHITECTURE.md` has been folded in here.)

## Commands

Everything needs a static server on the repo root — `file://` breaks ES modules and
`mountNativeFS`, and the tests all drive a real browser.

```bash
python3 -m http.server 8765 &      # required for every command below
npm --prefix tools install         # puppeteer-core lives under tools/

python3 tools/test_shims.py          # Python shim logic, no browser (~1s)
python3 tools/test_shims.py --sync   # ...in JSPI mode; BOTH must pass

node tools/drive.mjs                             # full notebook, real GPU (stages tools/test_image.tif)
node tools/drive.mjs --swiftshader               # ...on a machine with no usable GPU (WebGPU on the CPU, ~10 min)
node tools/test_agent.mjs                        # agent harness, all providers
node tools/test_agent.mjs --only gemini          # one of: anthropic, gemini, openai

cd tools && npm run serve            # same static server, via the npm script
```

The npm dev tooling (and `node_modules`) lives under `tools/` — this repo is served
as-is and has no build step, so the package files are colocated with the only things
that use them. Bare imports in `tools/*.mjs` resolve from `tools/node_modules`
regardless of the cwd you invoke them from.

`tools/notebook.mjs` is the headless-notebook layer both browser drivers share: launch Chrome
with the WebGPU flags, boot `notebook.html`, stage files, append cells, run all, read cells,
figures, and produced files back. `tools/runner.mjs` is the email agent's compute (see
`email-agent/README.md`). It claims jobs from the email-agent Worker, runs each in a fresh tab
through that same layer, and reports back. It launches Chrome with `protocolTimeout: 0`: a
cell that holds the main thread blocks every DevTools call, and puppeteer's default 180 s
per call would otherwise kill a run that was fine.

Useful flags: `drive.mjs --headful --logs --keep` (`--keep` preserves the cells in
localStorage instead of resetting to the seed script). Both browser drivers take
`CHROME_PATH`, and pass `--enable-unsafe-webgpu --use-angle=metal` — without those,
headless Chrome boots the page but `navigator.gpu` is missing and every model call
fails.

There is no build step, no bundler, and no lint config. Source is served as-is.

**Regression fixture.** `drive.mjs` stages `tools/test_image.tif` by default; override
it with `--stage <path>` for a different image. The documented baseline on
`test_image.tif`: 190 cells, 183 nuclei, 173 kept, a rendered DataFrame, one figure, and
two output files. Cold start is ~5 MB with no weights resident — if that number jumps,
something started loading eagerly again.

## Architecture

Four layers, each of which only knows about the one below it.

**Engines** (`vendor/webgpu-cellseg/src/{cellpose,stardist,instanseg}.js`) — hand-written WGSL, no ML
framework. Uniform shape: `create()` → `loadWeights()` → `segmentImage() → {labels:
Int32Array, timings}`. Treat these as near-frozen; they're validated against desktop
references.

**`src/registry.js` is the fusion point** and where most feature work belongs. It is the
only place that knows what models exist, where weights live, what input layout each
wants, and what each is good for. Everything above addresses models by string id.
Adding a model there teaches the Python shims, `ptero.models.list()`, *and* the agent's
system prompt at once — the prompt's routing table is generated from `catalogue()`, not
hand-written. `src/gpu.js` hands all engines one shared `GPUDevice`.

**Kernel + shims** (`src/kernel.js`, `src/shims/`) — Pyodide, and fake `cellpose` /
`stardist` / `instanseg` / `csbdeep` packages that imitate their upstream APIs so that
code copied from real docs (or written by an LLM from its priors) runs unmodified.
They're real importable packages written into the Pyodide FS and put on `sys.path`, not
`sys.modules` injection, so nothing executes until a cell imports one. `ptero.py` is the
discovery surface the imitations can't provide.

**Notebook + agent** (`src/notebook/`, `src/agent/`) — the cell model is a plain array
we own precisely because the agent acts on it directly; `push_to_ui` calls `insertCell`
rather than faking a click.

Layout:

```
notebook.html              the notebook + assistant pane
  src/notebook/ui.js         chrome: theme, workspace, toolbar, boot
  src/notebook/cells.js      the cell model + CodeMirror + run loop
  src/kernel.js              Pyodide, shim install, the JS↔Python bridge
    src/shims/*.py             fake cellpose / stardist / instanseg / csbdeep + ptero
  src/registry.js            model catalogue, lazy weights, uniform segment()
    src/gpu.js                 one shared GPUDevice
    src/style.css              all CSS
  src/agent/                 llm · agent · tools · prompt · chat · scratch(+worker)
tools/                     headless drivers + npm dev deps (package.json lives here)
  tools/notebook.mjs         drive notebook.html headlessly (shared by drive.mjs, runner.mjs)
  tools/runner.mjs           runs email-agent jobs in the notebook
email-agent/               the email front end: a Cloudflare Worker (own package.json, own README)
```

### Engines are vendored, not ours to edit

They live in `vendor/webgpu-cellseg/` and come from the `webgpu-cellseg` repo, which
holds the PyTorch reference dumps and the fidelity harnesses that establish their
numerics. Fix engine bugs and do kernel work *there*, then re-sync —
`vendor/webgpu-cellseg/VENDOR.md` has the procedure and explains what ptero deliberately
supplies instead (weights, and the shared device). Editing them here means the next
re-sync silently reverts you. The engines' own performance record — why the conv kernel
is shaped the way it is, and which levers are already pulled — is upstream in that
repo's `docs/PHASE0.md` and `PHASE1.md`. Read those before anything performance-related.

**One thing in `src/gpu.js` is load-bearing for the vendored kernel**:
`maxComputeWorkgroupStorageSize` must be requested at the adapter maximum. The conv
kernel stages several input channels per barrier round and does not fit in WebGPU's
16 KB default — it fails pipeline creation outright, which surfaces as an unrelated
`getBindGroupLayout` error because pipeline limits fail through *asynchronous*
validation. Keep it in step with `vendor/webgpu-cellseg/src/device.js`.

### Sync vs `await` — invisible above the shims

The GPU boundary is async, so shim methods are written `async def`. Pyodide's `run_sync`
(JS Promise Integration) makes them callable synchronously, which is what upstream
fidelity requires. `src/shims/_ptero_autoawait.py` rewrites every cell's source via an
AST transform before it runs, wrapping calls in a no-op pass-through that also resolves
coroutines — so cell and agent code never writes `await` and reads exactly like the real
upstream docs. The one scope limit: `await` is illegal inside a plain `def`/`lambda`, so
a user-defined synchronous helper that consumes a shim result inline still needs an
explicit `await` on a browser without JSPI.

### The shims imitate upstream deliberately

Deviations are documented in each module's docstring and repeated in the agent's system
prompt. The important ones:

- `Cellpose.eval` returns 4 values, `CellposeModel.eval` returns 3. **These are not the
  same class.** An earlier shim aliased them, which silently breaks correct upstream
  unpacking.
- Cellpose `flows`/`styles` are None; there is no SizeModel, so always pass a `diameter`.
- InstanSeg returns numpy, not torch.
- StarDist `details` has only the object count, not polygons.

### The agent

One agent, one conversation, no orchestration. Two hard constraints are enforced in code
and asserted in `tools/test_agent.mjs`:

- **Recursion cap** — max 3 consecutive failed scratch tests, then a re-prompt with
  `tools: []` so the model cannot keep grinding.
- **Context protection** — no traceback or failed draft ever reaches the chat
  transcript.

Both are load-bearing product behaviour, not defensive coding.

**The scratch kernel** is a Worker with mocked segmentation (`src/shims/_ptero_mock.py`).
It exists to catch shape and API errors, not to produce real object counts — the system
prompt tells the agent never to tune a threshold against it. It reads the workspace's
real files read-only-by-construction: `scratch-worker.js` copies file bytes into the
worker's own in-memory FS rather than mounting the directory handle, because the browser
shares one permission grant across every handle referencing a directory.

### Providers and the API key

`src/agent/llm.js` supports Anthropic, Google Gemini, and an OpenAI-compatible adapter
(default DeepInfra, base URL overridable for a local server). The conversation format
above `llm.js` is Anthropic's; Gemini and OpenAI translate both ways so `agent.js` never
learns which is in use. Load-bearing translation details:

- Gemini rejects unknown JSON Schema keywords, rejects an empty `parameters` object,
  keys tool results by function *name* rather than id, and can reject consecutive
  same-role turns — all handled in `geminiContents`/`geminiSchema`. Any new tool schema
  goes through that translation, so run `test_agent.mjs --only gemini` after touching
  `src/agent/tools.js`.
- OpenAI has no `tool_result` block — each becomes its own `role: "tool"` message keyed
  by `tool_call_id`, so one Anthropic `user` message with several results expands into
  several OpenAI messages.
- OpenAI streams tool-call arguments as fragments of a JSON *string*, indexed by
  position rather than id.

Model lists are fetched from the provider with the user's key rather than hardcoded.
Keys and model choices are stored per provider; keys saved under older single-provider
names are migrated on first load. The key is readable by anything on this origin, so:
keys are read from `localStorage` at call time and never parked on `window`, and pandas
`_repr_html_` output is sanitised into inert nodes before it reaches the DOM
(`cells.js`) — the previous version did `innerHTML = html` directly.

## Things that will waste your time if you don't know them

**`can_run_sync()` describes the current Python stack, not the browser.** It returns
`true` from `runPythonAsync` and `false` from `runPython` in the same page. Never cache
it — `maybe_sync` checks per call. This matters only if you're touching
`_ptero_bridge.py`/`_ptero_autoawait.py` themselves — cell and agent code never needs to
know the answer or write `await`.

**Never dunder-name a JS global called from a Python class method.** `js.__foo` inside a
method gets mangled to `js._ClassName__foo` and fails. The bridge globals are
`pteroSegment` / `pteroCatalogue` / `pteroEnvironment` for this reason.

**All cells share one Python global namespace.** That's the feature that makes variables
persist across cells, and the reason every kernel-level helper must scope its imports
*inside* the function — a cell doing `from skimage import io` otherwise clobbers stdlib
`io` out from under it.

**`tifffile` is not in the Pyodide lockfile.** It's pure Python, so
`loadPackagesFromImports` never fetches it; `kernel.js` micropip-installs it on demand.

**`tifffile` alone can't decode a compressed TIFF.** LZW/Deflate/JPEG/Zstd-compressed
TIFFs — most real microscopy files — need `imagecodecs`, which has no wasm wheel on
PyPI. `wheels/` carries a pruned, cross-compiled build (see `wheels/NOTICE` for exactly
which codecs it covers); `kernel.js` and `scratch-worker.js` micropip-install it by URL
in the same call as `tifffile`, so a cell never sees `tifffile` succeed at import only
to throw later on the first compressed file. Both kernels' install calls need to move
together if this ever changes.

**Nothing may load eagerly at boot.** Packages arrive via `loadPackagesFromImports` per
cell; weights load per model on first use. Adding an eager `loadPackage` or a top-level
weights fetch silently undoes the ~5 MB cold start.

**esm.sh + shared peer deps.** Import shared CodeMirror peers (e.g.
`@codemirror/state`) with the *same semver range the consuming package uses*, never a
chosen exact version — a second module instance fails CodeMirror's identity checks with
a cryptic "Unrecognized extension value".

**CSS: grid/flex items need `min-width: 0`, and `margin-inline: auto` on a grid item
disables stretch alignment** (it gets fit-content-sized instead). Both bugs put the
notebook column on top of the assistant pane and made the Send button unclickable while
looking perfectly normal. The load-bearing rules are commented as such in
`src/style.css`.

**Marshalling shapes that work.** numpy(f32) → JS via
`proxy.getBuffer('f32').data.slice()` then `.release()` the wasm-heap view. Multi-channel
images cross as one contiguous `[C,H,W]` array — a single proxy and one `getBuffer`,
rather than a list of proxies. JS `Int32Array` → numpy via `await`, which Pyodide
converts to a memoryview: `np.asarray(mv, dtype=np.int32).reshape(H, W)`.

**matplotlib.** Backend forced to `Agg` before any cell can import pyplot, and
`plt.show` patched to a no-op — under plain Agg it warns loudly in a way that reads as an
error. Figures are captured after every cell regardless.

## Conventions

Comments explain *why*, not what — several in this codebase document a bug that a
plausible-looking "cleanup" would reintroduce. Don't strip them.

## Deliberately not built

The click→`Cell_ID` annotation surface, offline vendoring of Pyodide and CodeMirror,
`.ipynb` import/export, multi-agent orchestration, Ctrl-S to a real file handle.

On the annotation surface, the design conclusion worth not re-litigating: **don't
screenshot the notebook and ask a vision model what was clicked.** The segmentation
output is already a label map, so a click resolves to an exact `Cell_ID` by array lookup
— free, deterministic, no vision model. Vision earns its keep only for the perceptual
judgement itself ("is this real signal or an artifact?"), where the right unit is a
*targeted crop* of the raw channels with the mask boundary overlaid.
