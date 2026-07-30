# ptero — architecture

Supersedes `PIPELINE_HANDOFF.md` and `harness_v1.md`. This is the *why*; read it before
changing architecture, not just before changing code.

## What this is

A local-first bio-image analysis tool. A scientist opens a page, points it at a folder
on their disk, and either writes Python in a notebook or describes what they want to an
agent that writes the Python for them. Segmentation runs on their GPU. Nothing is
uploaded anywhere except the agent's chat messages.

Three things were fused to get here, each of which already worked on its own:

1. **Three WebGPU segmentation engines** — Cellpose cyto3, StarDist (fluo + H&E),
   InstanSeg brightfield. Hand-written WGSL, no ML framework at inference.
2. **A Pyodide notebook** — real numpy/pandas/scikit-image/matplotlib in the tab, with
   a fake `cellpose` module routing `.eval()` to the GPU.
3. **An agent harness design** — a single "co-scientist" that drafts code, verifies it
   privately, and only then puts it in front of the user.

## Layers

```
index.html                 landing
notebook.html              the notebook + assistant pane
  src/notebook/ui.js         chrome: theme, workspace, toolbar, boot
  src/notebook/cells.js      the cell model + CodeMirror + run loop
  src/kernel.js              Pyodide, shim install, the JS↔Python bridge
    src/shims/*.py             fake cellpose / stardist / instanseg / csbdeep + ptero
  src/registry.js            model catalogue, lazy weights, uniform segment()
    src/gpu.js                 one shared GPUDevice
    src/{cellpose,stardist,instanseg}.js   the engines
  src/agent/                 llm · agent · tools · prompt · chat · scratch(+worker)
demo/*.html                four standalone model demos (still work, unchanged)
tools/drive.mjs            headless regression driver
```

### One device, three engines

Each engine's `static create()` used to call `requestAdapter()` itself. Fine when a page
used one engine; wasteful once a page can reach for all three. `gpu.js` memoises a
single device — all three asked for the identical limit set, so there was nothing to
reconcile. `Engine.load(url, { device })` lets the registry build on it.

### The registry is the fusion point

`registry.js` is the only place that knows what models exist, where their weights live,
what input layout each wants, and what each is good for. Everything above it addresses
models by string id. Adding a model there teaches the shims, the `ptero` catalogue and
the agent's system prompt about it at once — the prompt's routing table is *generated*
from `catalogue()`, not written out by hand.

Weights load lazily per model. A session that only uses StarDist never downloads
cyto3's 26 MB.

### The shims are real packages, not sys.modules injection

`src/shims/` holds importable packages written into the Pyodide FS at boot and put on
`sys.path`. `import cellpose` resolves through the normal import machinery, so nothing
executes — and numpy isn't loaded — until a cell actually imports one.

They imitate upstream deliberately: an LLM writing from its priors, and a human copying
from the real docs, both produce code that runs. Deviations are documented in each
module's docstring and repeated in the agent's system prompt. The important ones:

- `Cellpose.eval` returns 4 values, `CellposeModel.eval` returns 3. **These are not the
  same class.** An earlier shim aliased them, which silently breaks correct upstream
  unpacking.
- Cellpose `flows`/`styles` are None; there is no SizeModel, so always pass a `diameter`.
- InstanSeg returns numpy, not torch.
- StarDist `details` has only the object count, not polygons.

`ptero.py` is the discovery surface the imitations can't provide: `ptero.models.list()`
answers "what is installed here, and which should I use?" at runtime.

### Sync vs `await` — measured, not assumed, and now invisible

The GPU boundary is async, so shim methods are written `async def`. Pyodide's `run_sync`
(JS Promise Integration) makes them callable synchronously, which is what upstream
fidelity requires.

**`can_run_sync()` describes the current Python stack, not the browser.** It returns
True on a stack entered through `runPythonAsync` and False on a plain `runPython` stack
in the very same page — verified in Chrome, not inferred. So the check happens per call
inside `maybe_sync`, never cached at import time, and falls back to returning the
coroutine (caller must `await`) where JSPI is unavailable.

That used to leak into cell/agent code: whether generated code wrote `await model.eval(...)`
or plain `model.eval(...)` depended on which browser it was running in, and code copied
straight from the real upstream docs (never `await`, since the real APIs are synchronous)
would silently receive an unawaited coroutine instead of a label array on a browser
without JSPI. `src/shims/_ptero_autoawait.py` removes that: before a cell runs,
`_ptero_auto_await_source()` rewrites its source via an AST transform, wrapping every
call in `await _ptero_auto_await(...)` — a no-op pass-through for an ordinary value,
and a transparent resolve for a coroutine. `can_run_sync()` stays exactly as described
above, but it is now purely an implementation detail of `maybe_sync`; nothing above it
needs to know the answer or write `await`. The one deliberate scope limit: a
user-defined *synchronous* helper that consumes a shim result inline can't be reached
(`await` is illegal inside a plain `def`/`lambda`/`class` body), so those still need
`await` written by hand if the browser lacks JSPI — see the module's docstring.

### The agent

One agent, one conversation, no orchestration (per `harness_v1`). Two constraints are
enforced in code rather than hoped for:

- **Recursion cap.** Three consecutive failed scratch tests, then the loop re-prompts
  with `tools: []` — the model *cannot* keep grinding — and must explain the blockage in
  plain language.
- **Context protection.** Failed drafts and tracebacks stay in the model's working
  context, where they're needed, but never reach the chat transcript. The user sees
  "testing…", then either working code or a plain explanation.

Tools: `inspect_user_kernel`, `test_in_scratch`, `push_to_ui`, `read_cell_result`,
`ask_user`. The first and fourth are additions to harness_v1's three — without
`inspect_user_kernel` the mock shapes are guesses, and without `read_cell_result` the
agent is blind to the likeliest next event, the user pressing ▶ and getting an error.

**The scratch kernel is a Worker with mocked segmentation.** It catches shape
mismatches, wrong tuple arity, bad merges — bugs in the analysis code, which is what the
loop is for. It loads no weights and touches no GPU. `_ptero_mock.py` returns synthetic
label maps that are *plausible enough that downstream code really runs*: consecutive
labels from 1, sensible sizes, enough objects for a groupby to chew on. An empty mask
would let broken code pass by accident. It patches `_ptero_bridge.segment`, which covers
all three shims because they each call it through the module rather than binding it at
import.

The corollary, stated in the system prompt: **never tune a threshold against scratch
output.** Those object counts are fiction.

### Providers and the API key

Three providers are supported: Anthropic, Google Gemini, and an OpenAI-compatible
adapter, chosen in ⚙. The OpenAI-compatible adapter defaults to DeepInfra
(`docs.deepinfra.com`) but its Base URL field can point at any host speaking the OpenAI
Chat Completions wire format, including a local server — its API key is optional there,
since self-hosted servers typically don't check auth, unlike the other two providers.
All three allow browser-origin calls with a user-supplied key — Anthropic needs the
explicit `anthropic-dangerous-direct-browser-access` header, Gemini and DeepInfra serve
CORS by default (verified against the live endpoints, which answer a browser
`fetch`/preflight with a normal response rather than blocking it). No proxy either way,
which means a custom OpenAI-compatible host must serve CORS itself or the browser will
block the request.

The conversation format used above `llm.js` is Anthropic's — content blocks,
`tool_use`, `tool_result`. It is the most expressive of the three, so the Gemini and
OpenAI/DeepInfra adapters translate on the way out and normalise responses back into the
same shape; `agent.js` never learns which provider is in use. Load-bearing translation
details:

- Gemini rejects JSON Schema vocabulary it doesn't know, so tool schemas are stripped
  of `additionalProperties`/`$schema` recursively rather than maintained twice.
- A `parameters` object with no properties is rejected, so no-argument tools like
  `inspect_user_kernel` omit it entirely.
- Gemini keys tool results by function **name**, not by call id, so the id→name map is
  rebuilt while walking the conversation. Gemini also has no call ids of its own, so
  synthetic ones are minted for our own history to point at.
- OpenAI's wire format has no `tool_result` content block — each one becomes its own
  `role: "tool"` message keyed by `tool_call_id`, so a single Anthropic `user` message
  holding several tool results expands into several OpenAI messages.
- OpenAI streams tool-call arguments as fragments of a JSON *string*, indexed by
  position in the model's `tool_calls` array rather than by id (the id/name only arrive
  once, on that call's first chunk) — accumulated the same way Anthropic's
  `input_json_delta` fragments are.

Model lists are fetched from the provider with the user's key rather than hardcoded, so
they don't go stale. Keys and model choices are stored per provider, so switching back
and forth doesn't lose anything; keys saved under the older single-provider names are
migrated on first load.

The key is readable by anything on this origin. Two consequences are handled rather
than noted:

- keys are read from `localStorage` at call time and never parked on `window`;
- pandas `_repr_html_` output is **sanitised into inert nodes** before it reaches the
  DOM (`cells.js`). A DataFrame built from an untrusted file is otherwise a path to that
  key. The previous version did `innerHTML = html` directly.

## Gotchas that cost real time

**Never dunder-name a JS global called from a Python class method.** `js.__cpSegment`
inside a method gets mangled to `js._ClassName__cpSegment` → `AttributeError` pointing
nowhere useful. The bridge globals are `pteroSegment` / `pteroCatalogue` /
`pteroEnvironment` specifically because of this.

**All cells share one global namespace.** That's the feature that makes variables
persist across cells. It also means a cell doing `from skimage import io` clobbers the
stdlib `io` module out from under any kernel helper that imported it at module scope.
Every kernel-level helper scopes its imports *inside* the function.

**`tifffile` is not in the Pyodide lockfile.** It's pure Python, so
`loadPackagesFromImports` will never fetch it, and `io.imread("x.tif")` fails on a fresh
kernel — the single most likely first thing to go wrong, TIFF being the microscopy
format. `kernel.js` micropip-installs it on demand when a cell touches skimage/tifffile.

**esm.sh + shared peer dependencies.** Never pin a shared CodeMirror peer (e.g.
`@codemirror/state`) to an independently-chosen exact version. `codemirror@6.0.1`
depends on `^6.0.0`; importing `@6.4.1` separately creates a *second* module instance,
and CodeMirror's extension system does identity checks — the symptom is a cryptic
"Unrecognized extension value in extension set". Import shared peers with the same
semver range the consuming package uses. Check with
`curl -s https://esm.sh/<pkg>@<ver> | head`.

**`margin-inline: auto` on a grid item disables stretch alignment.** The auto margins
absorb the free space, so the item is sized to fit-content instead of to its track. With
CodeMirror's long unwrapped lines that made `#main` ~890px wide inside a 400px column,
where it painted over the assistant pane and made the Send button unclickable while
looking perfectly normal. `width: 100%` restores track sizing; `max-width` still caps
and the auto margins still centre. `min-width: 0` on grid/flex items is needed for the
same family of reasons.

**Marshalling shapes that work.** numpy(f32) → JS via `proxy.getBuffer('f32').data.slice()`
then `.release()` the wasm-heap view. Multi-channel images cross as one contiguous
`[C,H,W]` array — a single proxy and one `getBuffer`, rather than a list of proxies. JS
`Int32Array` → numpy via `await`, which Pyodide converts to a memoryview:
`np.asarray(mv, dtype=np.int32).reshape(H, W)`.

**matplotlib.** Backend forced to `Agg` before any cell can import pyplot, and
`plt.show` patched to a no-op — under plain Agg it warns loudly in a way that reads as
an error. Figures are captured after every cell regardless, exactly as Jupyter's inline
backend does. The patch mutates the module attribute, not a name binding, so it survives
whatever a cell imports it as.

## Verification

```bash
python3 -m http.server 8765 &     # file:// breaks ES modules and mountNativeFS
npm install
node tools/drive.mjs --stage demo/images/Composite.tif
```

Expected on `Composite.tif`: 190 cells, 183 nuclei, 173 kept, a rendered DataFrame, one
figure, and `Cell_Measurements.csv` + `Filtered_Cell_Labels.tif`. Cold start is ~5 MB
with no weights resident.

The cell/nuclei counts now match the desktop PyTorch reference (190/183) rather than the
older browser figures (176/178) — the tiled-inference work closed that gap.

## Deliberately not built

The click→`Cell_ID` annotation surface, offline vendoring of Pyodide and CodeMirror,
`.ipynb` import/export, multi-agent orchestration, Ctrl-S to a real file handle.

On the annotation surface, the design conclusion worth not re-litigating: **don't
screenshot the notebook and ask a vision model what was clicked.** The segmentation
output is already a label map, so a click resolves to an exact `Cell_ID` by array lookup
— free, deterministic, no vision model. Vision earns its keep only for the perceptual
judgement itself ("is this real signal or an artifact?"), where the right unit is a
*targeted crop* of the raw channels with the mask boundary overlaid. Structured data
stays structured. Label maps are kept addressable by ID all the way to the renderer as
cheap insurance for this.
