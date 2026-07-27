# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Commands

Everything needs a static server on the repo root — `file://` breaks ES modules and
`mountNativeFS`, and the tests all drive a real browser.

```bash
python3 -m http.server 8765 &      # required for every command below
npm install                        # puppeteer-core, for the browser tests

python3 tools/test_shims.py          # Python shim logic, no browser (~1s)
python3 tools/test_shims.py --sync   # ...in JSPI mode; BOTH must pass

node tools/drive.mjs --stage demo/images/Composite.tif   # full notebook, real GPU
node tools/test_agent.mjs                                # agent harness, both providers
node tools/test_agent.mjs --only gemini                  # one provider

node tools/profile.mjs                                   # Phase 0 throughput accounting
node tools/profile.mjs --only composite --repeats 9      # one workload, more repeats
node tools/convbench.mjs --shapes 4 --per-shape          # conv kernel variants
```

Useful flags: `drive.mjs --headful --logs --keep` (`--keep` preserves the cells in
localStorage instead of resetting to the seed script). Both browser drivers take
`CHROME_PATH`, and pass `--enable-unsafe-webgpu --use-angle=metal` — without those,
headless Chrome boots the page but `navigator.gpu` is missing and every model call
fails.

There is no build step, no bundler, and no lint config. Source is served as-is.

**Expected results.** `drive.mjs` on `Composite.tif`: 190 cells, 183 nuclei, 173 kept,
a rendered DataFrame, one figure, and two output files. Cold start is ~5 MB with no
weights resident — if that number jumps, something started loading eagerly again.

## Architecture

Four layers, each of which only knows about the one below it.

**Engines** (`src/{cellpose,stardist,instanseg}.js`) — hand-written WGSL, no ML
framework. Uniform shape: `create()` → `loadWeights()` → `segmentImage() → {labels:
Int32Array, timings}`. Treat these as near-frozen; they're validated against desktop
references and the demo pages are their regression suite.

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

**`src/profile/` is measurement, not pipeline.** It attaches to an engine by substituting
its `_mkEncoder` hook for one that wraps every compute pass in timestamp queries, so no op
method knows profiling exists and the unprofiled path keeps its behaviour. Byte traffic
and FLOPs are *derived from the shaders* in `cost.js` rather than counted — WebGPU has no
DRAM or occupancy counters — which only works because the kernels are hand-written. The
labels each op passes to `beginComputePass` are the contract between the engines and
`parseLabel()`; change them together. Nothing here is imported unless a profiling run asks
for it.

**The conv kernel is generated** (`src/conv-kernel.js`), one pipeline per K. Its four
constants — BLK, TS, RBY/RBX, CB — are a *joint* measured optimum, interior on every axis,
because they trade against two shared budgets: registers (BLK·RBY·RBX accumulators) and
threadgroup memory (CB tiles). Raising any one of them loses, sometimes by half. Re-run
`tools/convbench.mjs` before changing any of them; don't reason about one in isolation. The kernel it
replaced is frozen in `src/profile/baseline-conv.js` as the benchmark's zero point; don't
"clean it up" into an import of the current kernel or the waterfall loses its reference
and the equivalence check compares the new kernel against itself.

`docs/ARCHITECTURE.md` has the reasoning behind all of this and a longer gotcha list.
Read it before changing architecture. `docs/PHASE0.md` is the throughput accounting and
its gate decision — read it before optimising anything, because it says which 0.2% of the
dispatches are not worth your time. `docs/PHASE1.md` is what came of it — 11× on the
conv, then 10× on the flow-consistency QC once that became the bottleneck, then a joint
retune of the conv's three budgets, for **9.72× end-to-end** with the label maps unchanged. Read it before optimising: it records which
levers are already pulled and which four are next.

**Both rebuilt hot paths are exactness-verified, and that is the bar.** The conv is
bit-identical to its predecessor; the GPU flow QC differs on 0 of 1.2M pixels. Neither was
accepted on a tolerance — the flow QC in particular decides how many masks survive a
threshold, so its diffusion runs in f32 but the normalisation and per-mask error stay on
the CPU in f64 deliberately. Any change here gets the same treatment: run both
implementations on the same input and diff the label maps.

## Things that will waste your time if you don't know them

**`can_run_sync()` describes the current Python stack, not the browser.** It returns
`true` from `runPythonAsync` and `false` from `runPython` in the same page. Never cache
it — `maybe_sync` checks per call. A boot-time probe on a `runPython` stack will tell
you JSPI is unavailable on a browser that has it.

**Never dunder-name a JS global called from a Python class method.** `js.__foo` inside a
method gets mangled to `js._ClassName__foo` and fails. The bridge globals are
`pteroSegment` / `pteroCatalogue` / `pteroEnvironment` for this reason.

**All cells share one Python global namespace.** That's the feature that makes variables
persist across cells, and the reason every kernel-level helper must scope its imports
*inside* the function — a cell doing `from skimage import io` otherwise clobbers stdlib
`io` out from under it.

**`tifffile` is not in the Pyodide lockfile.** It's pure Python, so
`loadPackagesFromImports` never fetches it; `kernel.js` micropip-installs it on demand.
Any new pure-Python dependency needs the same treatment.

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
looking perfectly normal. The load-bearing rules are commented as such in `style.css`.

**Provider adapters.** The conversation format above `src/agent/llm.js` is Anthropic's;
Gemini translates both ways so `agent.js` never learns which is in use. Gemini rejects
unknown JSON Schema keywords, rejects an empty `parameters` object, keys tool results by
function *name* rather than id, and can reject consecutive same-role turns — all handled
in `geminiContents`/`geminiSchema`. Any new tool schema goes through that translation,
so run `test_agent.mjs --only gemini` after touching `src/agent/tools.js`.

## Conventions

Comments explain *why*, not what — several in this codebase document a bug that a
plausible-looking "cleanup" would reintroduce. Don't strip them.

The agent's two hard constraints are enforced in code and asserted in
`tools/test_agent.mjs`: **max 3 consecutive failed scratch tests**, then a re-prompt with
tools stripped; and **no traceback or failed draft ever reaches the chat transcript**.
Both are load-bearing product behaviour, not defensive coding.

The scratch kernel mocks segmentation on purpose (`src/shims/_ptero_mock.py`). It exists
to catch shape and API errors, not to produce real object counts — the system prompt
tells the agent never to tune a threshold against it.
