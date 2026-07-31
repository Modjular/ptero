# The board, an AI stats tool, and a per-session modality store

*(Design doc from a planning session — not yet implemented. Written so a fresh session
can pick this up without the original conversation's context. Revised once already: the
original draft designed Feature 1 as a single-image pan/zoom "human viewer" panel with
its own arbitrary-array-to-PNG rendering path. That's been replaced below with a
persistent multi-image scratch canvas — "the board" — driven entirely by images the
notebook already produces, after a second planning pass. If you're re-reading this after
more time has passed, diff this file against git history before trusting the "current
state" facts at the bottom — they were re-verified against the repo once, at revision
time, not continuously.)*

## Context

Three independent ergonomic gaps in ptero today:

- **Images have nowhere to live except inline in whichever cell produced them.** Once a
  second or third figure exists there's no way to see two side by side, come back to one
  from three cells ago, or mark it up — `plt.show()` output is print-once, not a
  workspace. This plan adds **the board**: a persistent, freeform surface the user sends
  images to on purpose, where they can be arranged, kept in view across the whole
  session, and drawn on. It is explicitly **not** the star of the show — no `ptero.show()`
  command, no auto-population, no becoming a second notebook. The user populates it by
  hand, one image at a time, the same way you'd pin a printout to a corkboard.
- **The agent never sees what it produces** — only shape/dtype metadata via
  `inspect_user_kernel`/`inspect_file`. This plan adds a second, independent tool
  returning *structured numeric stats* (histogram, label count, size distribution,
  border-touching count) about a named array, so the agent can sanity-check its own
  claims before making them. Deliberately **not** real pixels/vision — research
  confirmed that would require real per-provider work in `llm.js` (below); that's a
  separate future step, not part of this plan.
- **A small per-session fact store** (just `modality` for now — see the scope note under
  Feature 3) that the agent populates via `ask_user` + a new tiny tool once it learns the
  answer, so it doesn't have to re-derive or re-ask something it — or the user — already
  said this session. Scoped to the current workspace/session, reset on reload; never
  persisted across browser sessions, since a stale guess from a different folder is worse
  than asking again.

These three are independent of each other — the board doesn't touch the kernel at all,
`array_stats` doesn't touch the board, and the modality store doesn't currently feed
either (see the Feature 3 scope note). Build order: **Feature 3 → Feature 2 → Feature 1**
— cheapest and lowest-risk first, UI risk last. This order is a holdover from the
original draft; it's arguably even more true now that Feature 1 carries zero Python risk,
only DOM/CSS risk.

---

## Feature 3 — per-session modality store

**Scope note (changed from the original draft):** the original design stored
`{ modality, channelAxis }` — `channelAxis` existed solely to feed Feature 1's old
arbitrary-array render heuristic (`choose_channel_axis`). That heuristic is cut (see
Feature 1's design-principles note). `channelAxis` has no remaining consumer anywhere in
this plan, so it's dropped rather than kept "for later" — this codebase's own convention
(see `CLAUDE.md`: "Don't design for hypothetical future requirements") argues against
storing a field nothing reads. `modality` stays: the agent can plausibly use it when
reasoning about which model to suggest (`ptero.py`'s `_Models.suggest(stain, target,
modality)`, confirmed at `src/shims/ptero.py:36-61`) without re-asking mid-conversation.

**`src/kernel.js`** — add near the other module-level session state (`pyodide`,
`mplReady`), right after the `isReady()`/`raw()` pair (currently lines 75-76):

```js
let sessionInfo = { modality: null };
export function getSessionInfo() { return { ...sessionInfo }; }
export function setSessionInfo(patch) {
  if (patch.modality !== undefined) sessionInfo.modality = patch.modality;
}
export function resetSessionInfo() { sessionInfo = { modality: null }; }
```

Call `resetSessionInfo()` at the top of `mountWorkspace(handle)` (currently line 421 —
shifted from the original draft's "~340" because `describeFile()`/`inspect_file` was
added to this file since) — a new/switched folder invalidates any prior guess about
what's in it. A full page reload already resets the plain module variable, so no other
reset point is needed.

**`src/agent/tools.js`** — new `SCHEMAS` entry, placed right after `ask_user` (currently
the last entry, ~line 108-126 — the two are meant to be used as a pair: ask, then
record):

```js
{
  name: "set_image_info",
  description:
    "Remember the modality of the image(s) the user is working with — e.g. 'H&E', " +
    "'fluorescence', 'brightfield' — for the rest of this session. Call this right " +
    "after ask_user gives you the answer, so you don't have to ask again. Do not call " +
    "this to guess.",
  input_schema: {
    type: "object",
    properties: {
      modality: { type: "string", description: "e.g. 'H&E', 'fluorescence', 'brightfield'" },
    },
    required: ["modality"],
  },
},
```

`runTool` case (`src/agent/tools.js`'s `switch`, currently lines 137-242 — add after the
`ask_user` case):

```js
case "set_image_info": {
  if (!input.modality?.trim()) throw new Error("set_image_info needs a modality");
  kernel.setSessionInfo({ modality: input.modality });
  ui.note("noted image info");
  return `Recorded modality=${input.modality}.`;
}
```

Also extend the existing `inspect_user_kernel` case (currently lines 139-158) to append
`kernel.getSessionInfo()`'s `modality` to its returned string, if set. Free — no new tool
call — and it means the agent's habitual first move surfaces whatever it already learned.

**Test** — extend `tools/test_agent.mjs`'s `WORKING(enc)` fixture (defined at line 93)
with one more step after the `ask_user` response: `{ id: "t2b", name: "set_image_info",
input: { modality: "fluorescence" } }`, plus a `check(...)` (helper defined at line 26)
per provider confirming the schema survives Gemini/OpenAI translation and the
confirmation text appears later in `sent`.

---

## Feature 2 — AI stats tool: `array_stats` (stats only, no pixels)

**Design decision, unchanged from the original draft: `kind` (`"image"` | `"labels"`) is
a required argument, not auto-detected.** The agent already knows what it just computed
(it wrote the code); asking it to say so is less work than a heuristic that can silently
guess wrong, which is worse than no sanity check at all. This also reuses the exact
`kind: ["image", "labels"]` enum already taught to the model via `test_in_scratch`'s
`vars[].kind` (`src/agent/tools.js`, currently lines 64-65) — one vocabulary, not two.

**New file `src/shims/_ptero_stats.py`** — pure numpy, no `js` bridge dependency,
importable under plain CPython the same way `_ptero_bridge.py` already is, via
`tools/test_shims.py`'s `sys.path.insert(0, str(SHIMS))` (confirmed at line 71).

**Scope note (changed from the original draft):** this file no longer needs
`choose_channel_axis` — that function existed only to serve Feature 1's old
arbitrary-array render path, which is cut. `array_stats` itself never needed it; it
computes stats directly off the array it's given, no channel reasoning involved. So this
file now contains exactly one function.

```python
def array_stats(arr, kind):
    """arr: numpy array. kind: 'image' or 'labels'. Returns a JSON-serializable dict.
    Every numpy scalar is cast to plain float/int before returning — json.dumps() can't
    serialize np.float64/np.int64 directly, same discipline _ptero_describe follows."""
```

- **`kind == "image"`**: `min`, `max`, `mean`, `std`, percentiles `[1, 50, 99]`, and a
  32-bin histogram (`np.histogram(arr, bins=32)` → `{"counts": [...], "edges": [...]}`).
- **`kind == "labels"`**: guard `arr.max() == 0` explicitly (empty case) before
  computing. Otherwise: `n_labels = int(arr.max())` (labels are consecutive from 1 — this
  codebase's own convention, confirmed at `src/notebook/ui.js`'s seed script:
  `f"{cell_masks.max()} cells, {nuc_masks.max()} nuclei"`, line 33). `sizes` via
  `np.bincount(arr.ravel())[1:]` (drop background) → `min/max/mean/median`.
  `border_touching`: for 2-D arrays, count of distinct nonzero labels touching any edge
  row/column; for other ndim, omit the field and add a `"note"` explaining it's only
  computed for 2-D label maps rather than guessing.

**`src/kernel.js`** — add `"_ptero_stats.py"` to `SHIM_FILES` (currently lines 16-27,
alongside `_ptero_bridge.py`). New export near `describeGlobals()` (currently lines
290-314 — shifted from the original draft's "~283" because `describeFile()` was added
between them):

```js
export async function arrayStats(name, kind) {
  if (!pyodide) throw new Error("kernel not booted");
  await pyodide.loadPackage("numpy");   // no-op if already loaded — same on-demand
                                          // discipline as ensureTifffile() (line 169)
                                          // and ensurePillow() (line 178)
  const json = pyodide.runPython(String.raw`
def _ptero_array_stats(name, kind):
    import json
    from _ptero_stats import array_stats
    if name not in globals():
        return json.dumps({'error': f"no variable named {name!r} in the kernel"})
    try:
        return json.dumps(array_stats(globals()[name], kind))
    except Exception as e:
        return json.dumps({'error': str(e)})
_ptero_array_stats(${JSON.stringify(name)}, ${JSON.stringify(kind)})
`);
  return JSON.parse(json);
}
```

**`src/agent/tools.js`** — new `SCHEMAS` entry, placed after `inspect_file` (currently
lines 25-40 — same "look, don't guess" family as `inspect_user_kernel`/`inspect_file`):

```js
{
  name: "array_stats",
  description:
    "Get numeric summary stats for a named array — a histogram for intensity data, or " +
    "label count / size distribution / border-touching count for a label map. Use this " +
    "to sanity-check a claim before making it, instead of guessing. Does not show you " +
    "the actual pixels.",
  input_schema: {
    type: "object",
    properties: {
      name: { type: "string", description: "Variable name in the user's kernel." },
      kind: { type: "string", enum: ["image", "labels"] },
    },
    required: ["name", "kind"],
  },
},
```

`runTool` case:

```js
case "array_stats": {
  if (!input.name?.trim()) throw new Error("array_stats needs a variable name");
  ui.note(`checking ${input.name}…`);
  const stats = await kernel.arrayStats(input.name, input.kind);
  if (stats.error) throw new Error(stats.error);
  return `Stats for ${input.name} (${input.kind}):\n${JSON.stringify(stats, null, 2)}`;
}
```

**No changes needed to `src/agent/agent.js` or `src/agent/llm.js`** — confirmed by
reading `llm.js`: every provider adapter treats `tool_result.content` as a string
already (Gemini and OpenAI adapters explicitly `String()`-coerce it), and `agent.js`
pushes whatever `runTool` returns straight into `content` unwrapped
(`src/agent/agent.js`, tool-result handling around lines 71-85). A JSON-stringified
stats blob is just another string.

**Test** —
- `tools/test_shims.py`: new section importing `_ptero_stats`. Cover: `kind="image"` on
  a small synthetic array (histogram sums to `arr.size`); `kind="labels"` on a hand-built
  map with two clean objects and one border-touching object; an empty label map
  (`n_labels: 0`, no exception); a 3-D `kind="labels"` array (border_touching omitted +
  `note`, not a crash); confirm every result round-trips through `json.dumps` without a
  numpy-scalar `TypeError`. Run both `python3 tools/test_shims.py` and `--sync` — this
  module is synchronous pure numpy, touches no JSPI/`maybe_sync` path, so both should
  pass identically; the two-mode run is just this repo's standing convention.
- `tools/test_agent.mjs`: extend the `WORKING(enc)` fixture (line 93) with one more
  tool-call step — `array_stats` on the labels variable — plus per-provider
  schema-translation checks alongside the existing ones (helper `check` at line 26).

---

## Feature 1 — the board: a persistent scratch canvas for images

**Renamed from "human viewer" in the original draft.** Not a viewer for arbitrary kernel
variables anymore — a place the user *sends* images they already have. This removes the
old draft's entire arbitrary-array-to-PNG rendering path (`renderArrayPreview`,
`_ptero_render_array`, `choose_channel_axis`, the channel-axis guessing heuristic) and
with it every planned change to `src/kernel.js` for this feature — **the board needs zero
Python and zero kernel.js changes.** It only ever handles PNG data URIs the notebook
already produced.

**Design principles, stated up front because the biggest risk here is scope creep into
"an artifact" (the user's own words) instead of a scratch surface:**
- No `ptero.show()` or any other programmatic way to populate it. The agent has no tool
  for this in this plan — the only path onto the board is the user clicking a button on
  an image they can already see.
- No resize handles, no color picker, no shape library, no layers panel, no
  export/share button, no undo history beyond whatever the browser gives for free. If
  more is wanted later, add it later.
- Annotation exists to answer one concrete need the user named directly: circling or
  freehand-marking a region that needs attention (their example: a nucleus that wasn't
  segmented properly) — not general-purpose drawing.

### Data flow: how an image gets onto the board

`src/notebook/cells.js`'s `renderOutput()` (currently lines 216-236) already turns every
`figs` entry into an `<img class="cellout-fig">`, currently at lines 224-229:

```js
for (const b64 of figs || []) {
  const img = document.createElement("img");
  img.className = "cellout-fig";
  img.src = `data:image/png;base64,${b64}`;
  cell.outEl.appendChild(img);
}
```

Change: wrap each image and add a small hover button, following the exact "controls
appear on hover, docked to the element's edge" convention `.cellctrls` already
establishes for cell run/move/delete buttons (lines 84-96, 210-218 in `style.css`):

```js
for (const b64 of figs || []) {
  const wrap = document.createElement("div");
  wrap.className = "cellout-fig-wrap";
  const img = document.createElement("img");
  img.className = "cellout-fig";
  img.src = `data:image/png;base64,${b64}`;
  const sendBtn = document.createElement("button");
  sendBtn.className = "fig-to-board btn-subtle";
  sendBtn.type = "button";
  sendBtn.title = "Send to board";
  sendBtn.textContent = "⇱";
  sendBtn.addEventListener("click", () => onSendToBoard?.(img.src));
  wrap.append(img, sendBtn);
  cell.outEl.appendChild(wrap);
}
```

**`cells.js` does not import `board.js` directly.** Same "expose a hook, wire it once in
`ui.js`'s `start()`" shape `setRunHooks({ before, after })` already establishes (lines
241-244) — add a sibling:

```js
let onSendToBoard = null;
export function setOutputActions({ sendToBoard } = {}) { onSendToBoard = sendToBoard; }
```

`src/notebook/ui.js`'s `start()` (currently lines 166-171) gets one more call next to
`cells.setRunHooks(...)`:

```js
cells.setOutputActions({ sendToBoard: board.addImage });
```

This keeps `cells.js`'s only import as `kernel.js`, unchanged — `board.js` stays a
sibling UI module `ui.js` wires together, not a new dependency of the cell-output layer.

### `src/notebook/board.js` — new module, structurally parallel to `cells.js`

In-memory model — a flat array of tiles, two kinds:

```js
// { id, kind: 'image', src, x, y, marks: [...] }
// { id, kind: 'text',  text, x, y }
const tiles = [];
```

`marks` on an image tile is a plain array of `{ type: 'path', d }` (freehand — an SVG
path string) or `{ type: 'rect'|'ellipse', x, y, w, h }` (marquee) — stored as data, not
DOM, so they save/restore through the same JSON round-trip as everything else and the
SVG overlay is regenerated from that array on load, never persisted as markup.

- **`addImage(src)`** — push a new image tile at a cascading default position (offset
  from the last placed tile; wraps back to a corner once it walks off the visible area),
  render it, save. This is the sole entry point `cells.js`'s new button calls.
- **Drag to reposition** — `mousedown` on a tile, track `mousemove`/`mouseup`, update
  `tile.x/y`, apply via one `style.transform: translate(...)` — the same mouse-event
  shape the original draft's pan/zoom sketch already used, just moving a whole tile
  instead of panning one image.
- **No free resize for v1** — fixed max-width thumbnails, natural aspect ratio; a
  click-to-focus toggle (a larger CSS-scaled view) covers "I need to see this bigger"
  without building 8-handle resize + aspect-lock, which is real complexity for a feature
  meant to stay minimal.
- **Two annotation tools**, toggled from a small per-tile toolbar (same hover-reveal
  convention as `.cellctrls`): a **pen** (freehand — traces a path from
  `mousedown`→`mousemove`) and a **marquee** (drag a rectangle/ellipse to circle a
  region — the "flag this nucleus" case). Both render into an `<svg>` overlay
  absolutely positioned over the tile's `<img>`, same size, `pointer-events` enabled only
  while a tool is active so dragging the tile itself still works the rest of the time.
  Single fixed accent color (`--accent-danger`, since "flagging a problem" is the driving
  use case) — no picker.
- **Text notes** — a third tile kind, not an annotation on an image: `{ id, kind:'text',
  text, x, y }`, same drag mechanics, for a freestanding comment not tied to one image.
- **`refresh()`** — re-render all tiles from `tiles` into the board's DOM container; used
  once at mount time after loading saved state.

**Persistence** — mirrors `cells.js`'s own pattern exactly (`STORE_KEY`/`saveCells()`/
`scheduleSave()`, lines 37-64): `localStorage["ptero-board"]`, debounced save on every
drag/annotate/add, loaded once at mount. **Not** reset on workspace switch the way
`resetSessionInfo()` is (Feature 3) — the board holds already-embedded image content, not
references to files in a folder, so switching folders doesn't invalidate it the way a
modality guess is invalidated. Only an explicit "clear board" action (mirroring
`resetCells()`'s `confirm(...)` gate, `ui.js` line 185) empties it.

**`mount(el)`** — called once from `ui.js`'s `start()`, alongside `cells.mount(...)`.

### Where it lives — a responsive column/tab, replacing the original draft's floating overlay

The original draft chose a `position: fixed` bottom-right overlay specifically to avoid
CSS grid combinatorics. That reasoning no longer applies — the new requirement (tab on
small screens, persistent column to the left on bigger ones) means the grid problem has
to be solved head-on rather than dodged.

**`notebook.html`** — new top-level sibling `<aside id="board">`, structurally parallel to
`<aside id="chat">` (currently line 38), placed *before* `<div id="main">` in document
order (currently line 11) so it lands on the left under CSS Grid's default source-order
placement without extra `order` rules. New toolbar button `#boardtoggle` next to
`#chattoggle` (currently line 26), toggling `document.body.classList.toggle("board-open")`
— wired in `board.js`'s own module, independent of `chat.js`, the same way the original
draft kept its viewer toggle independent.

**`style.css`** — extends the existing chat grid rule rather than replacing it. Today,
`body.chat-open:has(#chat)` (lines 101-105) is the only grid state: 2 columns,
`minmax(0,1fr) minmax(320px,400px)`. Adding an independently-toggled left column means
four real states (chat/board each on or off):

```css
body:has(#board) { max-width: none; padding: 0; }
body.board-open:has(#board) {
  display: grid;
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr);
}
body.chat-open.board-open:has(#chat):has(#board) {
  grid-template-columns: minmax(280px, 360px) minmax(0, 1fr) minmax(320px, 400px);
}
body:not(.board-open) #board { display: none; }
#board, .board-tile { min-width: 0; }
```

The same two rules `style.css` already documents as fixes for a real shipped bug —
`min-width: 0` on grid/flex items (lines 106-114) and giving `#main` an explicit width
rather than relying on `margin-inline: auto` alone (lines 87-100) — apply here for the
identical reason: `#board`'s own content (full-size images) can overflow its track the
same way CodeMirror's long lines did. `--bg-viewport`/`--bg-viewport-drag`/`--bg-tile`
tokens (lines 12-14 dark, 51-53 light) are still mostly unused and fit the board's
background / drag-in-progress state well, same as the original draft suggested.

**Small screens: tabs, not simultaneous columns.** `style.css` today has zero responsive
breakpoints — this is the first one. Below a cutoff (768px is a reasonable, unvalidated
starting point; there's no existing convention in this file to match), collapse to one
visible pane at a time and repurpose the toggle buttons as tab selectors:

```css
@media (max-width: 768px) {
  body.chat-open:has(#chat), body.board-open:has(#board) { grid-template-columns: 1fr; }
  body.chat-open #main, body.board-open #main { display: none; }
  body.chat-open.board-open #chat { display: none; } /* only one non-main tab at a time */
}
```

This needs real hands-on testing (resize the window / use device toolbar) — reading the
CSS is not enough to trust it, same standing rule as everything else UI-facing in this
project.

**Zero footprint, more strongly than the original draft's claim**: the board never calls
into `kernel.js`, never touches `cells.saveCells()`/`localStorage["ptero-cells"]`, and no
cell source is ever read by this path. Its own state lives entirely under a separate
`localStorage["ptero-board"]` key.

**Test** —
- No Python and no agent-tool changes in this feature, so no `test_shims.py` or
  `test_agent.mjs` coverage applies here — unlike the original draft's
  `choose_channel_axis` unit tests, which no longer exist to test.
- **Manual browser check — required, not optional**: `python3 -m http.server 8765 &`,
  open `notebook.html`, run the seed script through the matplotlib cell, click "send to
  board" on the resulting figure → confirm it appears on the board at a sane default
  position; drag it; use the pen tool to mark a region, the marquee tool to circle
  another; add a text note; reload the page → confirm the board's layout and marks
  survive (persistence) while diffing `localStorage["ptero-cells"]` before/after confirms
  it is byte-identical (zero footprint); narrow the window past the responsive breakpoint
  and confirm tab behavior; open board + chat together at a wide width and confirm the
  three-column layout doesn't overlap or clip.

---

## Verification summary

| Step | Command | Covers |
|---|---|---|
| 1 | `python3 tools/test_shims.py` and `--sync` | `_ptero_stats.array_stats` |
| 2 | `node tools/test_agent.mjs` | `set_image_info`, `array_stats` tool schemas/wiring, all 3 providers |
| 3 | manual browser session per the procedure above | board: add/drag/annotate/persist/responsive layout |

All three must pass before considering this done — steps 1-2 are the automatable floor;
step 3 is the one `CLAUDE.md` explicitly says can't be skipped for a UI-facing change.

---

## Facts this plan relies on (re-verified against the repo directly during this revision)

- `src/kernel.js` (454 lines total as of this revision): `isReady()`/`raw()` at lines
  75-76 (Feature 3's insertion point). `MPL_SETUP` lines 38-69, defines
  `_pop_mpl_figures` (58-68). `run()` lines 233-283, returns `{ text, html, figs }`,
  `figs` is base64 PNG strings. `describeGlobals()` lines 290-314 (metadata-only,
  JSON round-trip). `describeFile()` lines 325-388 — backs the `inspect_file` tool,
  added to this codebase since the original draft was written; reads
  shape/dtype/columns straight from file headers (tifffile/Pillow lazy-open, `.npy`
  `mmap_mode`) without loading data into a kernel variable. `installBridge()`'s
  `pteroSegment` lines 82-109 — the one raw-bytes bridge, used only for feeding the
  WebGPU segmentation engines; not touched by anything in this plan.
  `ensureMatplotlib()` lines 194-203 — only activates `MPL_SETUP` for matplotlib a cell
  already imported; not called by Feature 1 anymore since the board never renders
  arbitrary arrays. `ensureTifffile()` lines 169-175 and `ensurePillow()` lines 178-182
  are both live precedents for "load a package on demand, not at boot" — Feature 2's
  `arrayStats()` follows the same shape with `pyodide.loadPackage("numpy")`.
  `SHIM_FILES`/`SHIM_ROOTS` lines 16-32. `mountWorkspace(handle)` at line 421 (Feature
  3's `resetSessionInfo()` call site).
- `src/agent/tools.js` (242 lines total): `SCHEMAS` array lines 14-127 — six tools today
  (`inspect_user_kernel`, `inspect_file`, `test_in_scratch`, `push_to_ui`,
  `read_cell_result`, `ask_user`); `inspect_file` (lines 25-40) is new since the original
  draft. `runTool(name, input, ui)` switch lines 137-242. Adding a tool = one array entry
  + one case, nothing else to register. `test_in_scratch`'s `vars[].kind` is
  `{ type: "string", enum: ["image", "labels"] }` at lines 64-65 — the vocabulary
  `array_stats` reuses.
- `src/agent/agent.js`: `runTool()`'s return goes straight into `{ type: "tool_result",
  content: out }` around lines 71-85, no wrapping/type-checking.
- `src/agent/llm.js`: all three provider adapters (Anthropic, Gemini, OpenAI/DeepInfra)
  treat `tool_result` content as text-only; Gemini/OpenAI explicitly `String()`-coerce
  it. Zero existing precedent for image/binary content blocks anywhere — confirms a
  stats-only tool needs no `llm.js` changes, and that a future real-pixel vision feature
  would need real per-adapter work there (explicitly out of scope for this plan).
- `src/notebook/cells.js` (314 lines total): `renderOutput()` lines 216-236, the fig loop
  specifically at lines 224-229 — this is Feature 1's sole integration point.
  `setRunHooks({ before, after })` lines 241-244, registered once in `ui.js`'s `start()`
  (lines 166-171); Feature 1 adds a sibling `setOutputActions` rather than overloading
  this hook, since it's a different concern (an output-affordance callback, not a
  run-lifecycle hook).
- `notebook.html` (96 lines total): `<body class="chat-open">` (line 10) — chat starts
  open by default, a static class in the markup, not something JS toggles at boot. Two
  top-level siblings under `<body>` today: `<div id="main">` (line 11) and `<aside
  id="chat">` (line 38), plus a normally-hidden `<dialog id="settings">` (line 51).
  Toolbar (lines 21-29) holds `#runall, #choosews, #refreshws, #resetcells, #chattoggle,
  #themetoggle, #stat`; `#addcell` lives separately in its own `.addcell-row` below
  `#cells` (line 34), not in the toolbar.
- `style.css` (355 lines total): `body:has(#chat)` (line 86) drops max-width/padding.
  `body.chat-open:has(#chat)` (lines 101-105) is the only grid state that exists today —
  2 columns. Two rules explicitly documented in comments as fixes for a real shipped bug:
  `#main`'s explicit `width: 100%` rather than relying on `margin-inline: auto` alone
  (lines 87-100), and `min-width: 0` on grid/flex items (lines 106-114) — both because
  omitting them let the notebook column overflow its track and cover the assistant pane,
  making Send unclickable while looking fine. `--bg-viewport`/`--bg-viewport-drag`/
  `--bg-tile` tokens exist at lines 12-14 (dark) and 51-53 (light), still almost entirely
  unused. **No `@media` breakpoints exist anywhere in this file** — the board's tab
  behavior on small screens is the first responsive rule in the codebase, so it needs
  real device-width testing rather than pattern-matching an existing convention.
- `src/agent/chat.js`: the existing toggle pattern — `#chattoggle` does
  `document.body.classList.toggle("chat-open")`; CSS reacts to the class; not persisted
  to localStorage. Feature 1's `#boardtoggle` mirrors this exactly with `"board-open"`.
- `src/notebook/ui.js` (226 lines total): `start()` (lines 166-219) is where
  `cells.mount(...)`, `cells.setRunHooks(...)`, and the toolbar's event listeners are all
  wired once; `board.mount(...)` and `cells.setOutputActions(...)` join that same
  sequence. `chat.js` is dynamically imported at line 214 so a session that never opens
  the assistant doesn't pay for it — `board.js` is small/dependency-free enough (no
  CodeMirror, no LLM client) that it doesn't need the same treatment; import it
  statically like `cells.js`.
- `src/shims/ptero.py`: `_Models.suggest(stain, target, modality)` at lines 36-61 — the
  stain/modality → model-id routing function that gives Feature 3's `modality` field a
  plausible future consumer, even though nothing in this plan calls it with the stored
  value yet.
- `tools/test_shims.py`: `sys.path.insert(0, str(SHIMS))` at line 71 makes any file under
  `src/shims/` directly importable under plain CPython.
- `tools/test_agent.mjs`: `WORKING(enc)` fixture at line 93; `check(name, ok, extra)`
  helper at line 26.
