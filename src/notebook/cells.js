// The cell model: an ordered array of { id, author, source, output } plus the
// CodeMirror views that edit them.
//
// This is deliberately a plain array we own rather than a notebook framework's
// internal state, because the agent acts on it directly — "insert a cell after cell 2
// and run it" is a function call into `insertCell`/`runCell`, not a synthetic click.
// Everything the agent needs is exported.
import { EditorView, basicSetup } from "https://esm.sh/codemirror@6.0.1";
// Pinned with the same "^6.0.0" range codemirror@6.0.1 itself uses for
// @codemirror/state, so esm.sh resolves both to the identical concrete build.
// Importing an unrelated exact version here (e.g. "@6.4.1") creates a second,
// incompatible copy of the module, and CodeMirror's extension system does identity
// checks — the symptom is a cryptic "Unrecognized extension value in extension set".
// Check what range a package expects with `curl -s https://esm.sh/<pkg>@<ver> | head`
// before adding anything else from the CodeMirror family.
import { Compartment } from "https://esm.sh/@codemirror/state@^6.0.0";
import { python } from "https://esm.sh/@codemirror/lang-python@6.1.6";
import { oneDark } from "https://esm.sh/@codemirror/theme-one-dark@6.1.2";

import * as kernel from "../kernel.js";

const themeCompartment = new Compartment();

let cellsEl = null;
let cellSeq = 0;
let darkTheme = true;
let onChange = () => {};       // fired whenever the cell list or a source changes

export const cells = [];

// ---- persistence ------------------------------------------------------------------
// Source only, never output: figures and tables are cheap to regenerate by re-running
// and would blow through localStorage's ~5-10 MB quota fast.
const STORE_KEY = "ptero-cells";
let saveTimer = null;

export function saveCells() {
  const payload = cells.map(c => ({ source: c.view.state.doc.toString(), author: c.author }));
  try {
    localStorage.setItem(STORE_KEY, JSON.stringify(payload));
  } catch (e) {
    console.warn("couldn't persist cells:", e.message);
  }
}

function scheduleSave() {
  clearTimeout(saveTimer);
  saveTimer = setTimeout(saveCells, 500);
}

export function loadSavedCells() {
  try {
    const arr = JSON.parse(localStorage.getItem(STORE_KEY));
    if (!Array.isArray(arr) || !arr.length) return null;
    // Tolerate the older format, which stored a bare array of source strings.
    return arr.map(e => (typeof e === "string" ? { source: e, author: "user" } : e))
              .filter(e => typeof e?.source === "string");
  } catch {
    return null;   // corrupt or missing — the notebook starts empty
  }
}

// ---- construction -----------------------------------------------------------------
function cmTheme() { return darkTheme ? oneDark : []; }

function buildCell({ source = "", author = "user" } = {}) {
  const id = ++cellSeq;
  const wrapEl = document.createElement("div");
  wrapEl.className = "cell";
  wrapEl.dataset.author = author;
  wrapEl.dataset.state = "idle";

  // Controls live docked to the cell edge and only appear on hover/focus — the
  // notebook should read as a document, not a control panel.
  const ctrls = document.createElement("div");
  ctrls.className = "cellctrls";
  const mk = (cls, text, title) => {
    const b = document.createElement("button");
    b.className = cls; b.textContent = text; b.title = title;
    b.type = "button";
    return b;
  };
  const runBtn = mk("cellrun", "▶", "Run cell (⇧⏎)");
  const upBtn = mk("cellmove btn-subtle", "↑", "Move up");
  const downBtn = mk("cellmove btn-subtle", "↓", "Move down");
  const delBtn = mk("celldel btn-subtle", "×", "Delete cell");
  ctrls.append(runBtn, upBtn, downBtn, delBtn);

  const editorHost = document.createElement("div");
  editorHost.className = "celleditor";
  const outEl = document.createElement("div");
  outEl.className = "cellout";
  // Status: elapsed time while running, then the run duration. The cell's colored
  // left border carries idle/running/done/error, so this never has to say it in words.
  const statusEl = document.createElement("div");
  statusEl.className = "cellstat";

  // The prompt is the cell's position in the document, not an execution count — it's
  // there so a person can say "cell 3" and mean something stable, which a Jupyter-style
  // run counter (reused across out-of-order re-runs) would not give them. renumber()
  // keeps it in sync whenever the list's order or length changes.
  const promptEl = document.createElement("div");
  promptEl.className = "cell-prompt";

  const bodyEl = document.createElement("div");
  bodyEl.className = "cell-body";
  bodyEl.append(ctrls, editorHost, outEl, statusEl);
  wrapEl.append(promptEl, bodyEl);

  const view = new EditorView({
    doc: source,
    extensions: [
      basicSetup,
      python(),
      themeCompartment.of(cmTheme()),
      EditorView.updateListener.of(u => { if (u.docChanged) scheduleSave(); }),
      // Shift-Enter runs, the way every notebook does. Keymap precedence puts this
      // ahead of basicSetup's default newline handling.
      EditorView.domEventHandlers({
        keydown(e) {
          if (e.key === "Enter" && e.shiftKey) { e.preventDefault(); runCell(cell); return true; }
          return false;
        },
      }),
    ],
    parent: editorHost,
  });

  const cell = { id, author, view, wrapEl, outEl, statusEl, promptEl, lastResult: null };
  runBtn.addEventListener("click", () => runCell(cell));
  delBtn.addEventListener("click", () => removeCell(cell));
  upBtn.addEventListener("click", () => moveCell(cell, -1));
  downBtn.addEventListener("click", () => moveCell(cell, +1));
  return cell;
}

/** Keep each cell's visible prompt in sync with its position in the document. */
function renumber() {
  cells.forEach((c, i) => { c.promptEl.textContent = `In [${i + 1}]`; });
}

/** Append a cell at the end. Returns the cell. */
export function appendCell(source = "", author = "user") {
  return insertCell(cells.length, source, author);
}

/**
 * Insert a cell at `index` (clamped). This is what the agent's push_to_ui calls.
 * Returns the cell.
 */
export function insertCell(index, source = "", author = "user") {
  const cell = buildCell({ source, author });
  const at = Math.max(0, Math.min(index, cells.length));
  cells.splice(at, 0, cell);
  cellsEl.insertBefore(cell.wrapEl, cellsEl.children[at] ?? null);
  renumber();
  saveCells();
  onChange();
  return cell;
}

/** Replace a cell's source in place, keeping its identity and position. */
export function setCellSource(cell, source) {
  cell.view.dispatch({
    changes: { from: 0, to: cell.view.state.doc.length, insert: source },
  });
  saveCells();
}

export function removeCell(cell) {
  const idx = cells.indexOf(cell);
  if (idx === -1) return;
  cells.splice(idx, 1);
  cell.view.destroy();
  cell.wrapEl.remove();
  renumber();
  saveCells();
  onChange();
}

function moveCell(cell, delta) {
  const from = cells.indexOf(cell);
  const to = from + delta;
  if (from === -1 || to < 0 || to >= cells.length) return;
  cells.splice(from, 1);
  cells.splice(to, 0, cell);
  cellsEl.insertBefore(cell.wrapEl, cellsEl.children[to] ?? null);
  renumber();
  saveCells();
  onChange();
}

// ---- output rendering --------------------------------------------------------------
// pandas' _repr_html_ is generated from data we did not write — a DataFrame built from
// an arbitrary file can carry markup in its cell values. This page holds an API key in
// localStorage, so that output is inserted as inert nodes rather than parsed as live
// HTML: the table structure is rebuilt from a detached parse, and script/style/event
// handlers are dropped on the way through.
const ALLOWED_TAGS = new Set(["TABLE", "THEAD", "TBODY", "TFOOT", "TR", "TH", "TD",
                              "DIV", "SPAN", "P", "B", "I", "EM", "STRONG", "BR",
                              "CODE", "PRE", "UL", "OL", "LI", "SMALL", "SUP", "SUB"]);

function sanitize(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");
  const out = document.createDocumentFragment();
  const clean = (src, dest) => {
    for (const node of src.childNodes) {
      if (node.nodeType === Node.TEXT_NODE) {
        dest.appendChild(document.createTextNode(node.nodeValue));
      } else if (node.nodeType === Node.ELEMENT_NODE && ALLOWED_TAGS.has(node.tagName)) {
        const el = document.createElement(node.tagName.toLowerCase());
        // Only structural table attributes survive; no style, no href, no on*.
        for (const attr of ["colspan", "rowspan"]) {
          if (node.hasAttribute(attr)) el.setAttribute(attr, node.getAttribute(attr));
        }
        clean(node, el);
        dest.appendChild(el);
      } else if (node.nodeType === Node.ELEMENT_NODE) {
        clean(node, dest);   // unknown wrapper: keep its text, drop the element
      }
    }
  };
  clean(doc.body, out);
  return out;
}

export function renderOutput(cell, { text, figs, html } = {}) {
  cell.outEl.replaceChildren();
  if (text) {
    const pre = document.createElement("pre");
    pre.className = "cellout-text";
    pre.textContent = text;
    cell.outEl.appendChild(pre);
  }
  for (const b64 of figs || []) {
    const img = document.createElement("img");
    img.className = "cellout-fig";
    img.src = `data:image/png;base64,${b64}`;
    cell.outEl.appendChild(img);
  }
  if (html) {
    const wrap = document.createElement("div");
    wrap.className = "cellout-html";
    wrap.appendChild(sanitize(html));
    cell.outEl.appendChild(wrap);
  }
}

// ---- running ------------------------------------------------------------------------
let beforeRun = async () => {};
let afterRun = async () => {};
export function setRunHooks({ before, after } = {}) {
  if (before) beforeRun = before;
  if (after) afterRun = after;
}

/**
 * Run one cell. Returns true on success. Never throws — a Python error is rendered
 * into the cell and recorded on `cell.lastResult` (which is what the agent's
 * read_cell_result reads).
 */
export async function runCell(cell) {
  if (!kernel.isReady()) {
    cell.statusEl.textContent = "kernel still booting…";
    return false;
  }
  await beforeRun();
  cell.wrapEl.dataset.state = "running";
  const t0 = performance.now();
  const tick = setInterval(() => {
    cell.statusEl.textContent = `${((performance.now() - t0) / 1000).toFixed(1)}s`;
  }, 100);

  let ok = true;
  try {
    const src = cell.view.state.doc.toString();
    const bundle = await kernel.run(src);
    renderOutput(cell, bundle);
    cell.lastResult = { ok: true, text: bundle.text, html: !!bundle.html, figs: bundle.figs.length };
    cell.wrapEl.dataset.state = "done";
  } catch (e) {
    const msg = String(e.message || e);
    renderOutput(cell, { text: msg });
    cell.lastResult = { ok: false, error: msg };
    cell.wrapEl.dataset.state = "error";
    console.error(e);
    ok = false;
  } finally {
    clearInterval(tick);
    const secs = (performance.now() - t0) / 1000;
    cell.statusEl.textContent = secs < 0.05 ? "" : `${secs.toFixed(1)}s`;
  }
  await afterRun();
  return ok;
}

/** Run every cell top to bottom, stopping at the first error. */
export async function runAll() {
  for (const cell of [...cells]) {
    if (!await runCell(cell)) return false;
  }
  return true;
}

// ---- theme + mount -------------------------------------------------------------------
export function applyTheme(dark) {
  darkTheme = dark;
  for (const cell of cells) {
    cell.view.dispatch({ effects: themeCompartment.reconfigure(cmTheme()) });
  }
}

export function mount(el, { dark = true, onChange: cb } = {}) {
  cellsEl = el;
  darkTheme = dark;
  if (cb) onChange = cb;
}

/** Scroll a cell into view and flash it — used when the agent writes into the notebook. */
export function highlight(cell) {
  cell.wrapEl.scrollIntoView({ behavior: "smooth", block: "center" });
  cell.wrapEl.classList.remove("cell-flash");
  void cell.wrapEl.offsetWidth;   // restart the animation
  cell.wrapEl.classList.add("cell-flash");
}
