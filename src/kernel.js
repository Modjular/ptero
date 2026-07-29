// The Python kernel: Pyodide, the shim packages, and the bridge to the WebGPU engines.
//
// Everything that has to know about Pyodide's quirks lives here, so the notebook UI
// above it deals only in { text, html, figs } display bundles and never touches a
// PyProxy. The agent's scratch worker reuses the same shim sources with a mocked
// bridge (see agent/scratch-worker.js).
import { segment, catalogue, environment } from "./registry.js";

const PYODIDE_VERSION = "0.28.0";
const PYODIDE_CDN = `https://cdn.jsdelivr.net/pyodide/v${PYODIDE_VERSION}/full/`;

// The fake libraries, written into the Pyodide filesystem as real importable packages
// rather than injected into sys.modules. That means `import cellpose` resolves through
// the ordinary import machinery — no boot-time execution, so nothing here costs
// anything until a cell actually imports one.
export const SHIM_FILES = [
  "_ptero_bridge.py",
  "_ptero_autoawait.py",
  "ptero.py",
  "cellpose/__init__.py",
  "cellpose/models.py",
  "stardist/__init__.py",
  "stardist/models.py",
  "csbdeep/__init__.py",
  "csbdeep/utils.py",
  "instanseg/__init__.py",
];
const SHIM_DIR = "/lib/ptero_shims";
// Importing any of these pulls in _ptero_bridge, which needs numpy. A cell can import
// a shim without importing numpy itself, and loadPackagesFromImports only sees the
// cell's own import lines — so match on these names and load numpy alongside.
const SHIM_ROOTS = ["cellpose", "stardist", "instanseg", "csbdeep", "ptero"];

// Force the plain PNG-producing Agg backend before any cell can import pyplot, and
// expose a helper that pops+encodes whatever figures a cell left open — the same
// "run the cell, then flush open figures as images" behaviour as Jupyter's inline
// backend.
const MPL_SETUP = String.raw`
import matplotlib
matplotlib.use('Agg')

# plt.show() has no interactive window to show under Agg, so it warns loudly by default
# ("FigureCanvasAgg is non-interactive, and thus cannot be shown") — harmless, but it
# looks like an error to someone without that context. Every cell's open figures are
# captured automatically after it runs (see _pop_mpl_figures below), the same "figures
# just appear" behaviour as Jupyter's inline backend — which likewise makes plt.show()
# a silent no-op. Match that, since plt.show() is deep matplotlib muscle memory and
# cells will keep calling it. Patching the module attribute (not a name binding) keeps
# it patched no matter what a cell imports it as.
import matplotlib.pyplot as _mpl_setup_plt
_mpl_setup_plt.show = lambda *a, **k: None
del _mpl_setup_plt

# NOTE: every cell shares one global namespace — that is what makes variables persist
# across cells — so this helper imports base64/io *locally*, not at module scope. A
# cell doing "from skimage import io" would otherwise clobber the stdlib io module out
# from under it.
def _pop_mpl_figures():
    import base64, io
    import matplotlib.pyplot as plt
    figs = []
    for num in plt.get_fignums():
        fig = plt.figure(num)
        buf = io.BytesIO()
        fig.savefig(buf, format='png', bbox_inches='tight', dpi=110)
        figs.append(base64.b64encode(buf.getvalue()).decode('ascii'))
    plt.close('all')
    return figs
`;

let pyodide = null;
let mplReady = false;
let onStdout = () => {};

export function isReady() { return pyodide !== null; }
export function raw() { return pyodide; }   // escape hatch for tools/drivers

// ---- the bridge ------------------------------------------------------------------
// Named without a leading dunder on purpose: a JS global referenced by name inside a
// Python *class method* gets mangled by Python's private-name rule (js.__foo becomes
// js._ClassName__foo) and fails with an AttributeError that points nowhere useful.
function installBridge() {
  // Python hands over one contiguous [C,H,W] float32 array — a single numpy proxy,
  // one getBuffer, no list-of-proxies marshalling — and gets back an Int32Array, which
  // Pyodide converts to a memoryview on the Python side.
  globalThis.pteroSegment = async (modelId, stackProxy, C, H, W, optsJson) => {
    const pb = stackProxy.getBuffer("f32");
    let flat;
    try {
      flat = pb.data.slice();     // own copy — the wasm heap view must not outlive this
    } finally {
      pb.release();
    }
    const n = H * W;
    const planes = [];
    for (let c = 0; c < C; c++) planes.push(flat.subarray(c * n, (c + 1) * n));

    const t0 = performance.now();
    const { labels, n: count, timings } = await segment(modelId, planes, H, W, JSON.parse(optsJson));
    console.log(`${modelId}: ${count} objects in ${(performance.now() - t0) | 0}ms ` +
                `(forward ${timings.forward | 0}ms)`);
    return labels;
  };

  // JSON strings rather than objects, so Python gets a str it can json.loads instead
  // of a JsProxy it would have to convert and destroy.
  globalThis.pteroCatalogue = () => JSON.stringify(catalogue());
  globalThis.pteroEnvironment = async () => JSON.stringify(await environment());
}

// ---- boot -------------------------------------------------------------------------
async function installShims(py) {
  const base = new URL("./shims/", import.meta.url).href;
  const sources = await Promise.all(
    SHIM_FILES.map(f => fetch(base + f).then(r => {
      if (!r.ok) throw new Error(`can't load shim ${f}: HTTP ${r.status}`);
      return r.text();
    }))
  );
  const mkdir = (p) => { try { py.FS.mkdir(p); } catch { /* exists */ } };
  mkdir("/lib");
  mkdir(SHIM_DIR);
  SHIM_FILES.forEach((f, i) => {
    const slash = f.lastIndexOf("/");
    if (slash > -1) mkdir(`${SHIM_DIR}/${f.slice(0, slash)}`);
    py.FS.writeFile(`${SHIM_DIR}/${f}`, sources[i]);
  });
  py.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(SHIM_DIR)})`);
}

/**
 * Boot Pyodide and make the kernel usable. Deliberately loads no science packages:
 * numpy/pandas/scikit-image/matplotlib arrive lazily, the first time a cell actually
 * imports them (see `run`). Combined with the registry's lazy weights, a cold start
 * downloads only the Pyodide core.
 */
export async function boot({ stdout } = {}) {
  if (pyodide) return pyodide;
  onStdout = stdout ?? (() => {});
  installBridge();

  if (!globalThis.loadPyodide) throw new Error("pyodide.js did not load");
  const py = await globalThis.loadPyodide({
    indexURL: PYODIDE_CDN,
    stdout: (m) => onStdout(m),
    stderr: (m) => onStdout(m),
  });
  await installShims(py);
  // Unconditional, unlike the segmentation shims (which load lazily on first import):
  // every cell needs the auto-await rewrite applied, including the very first one, which
  // may not import a ptero shim at all. `_ptero_auto_await` also has to land in the same
  // global namespace cells execute in, since the rewritten source calls it unqualified.
  py.runPython("from _ptero_autoawait import _ptero_auto_await_source, _ptero_auto_await");
  pyodide = py;
  globalThis.pyodide = py;   // for headless drivers / DevTools poking
  return py;
}

// ---- package loading --------------------------------------------------------------
const imports = (src, name) => new RegExp(`\\b(?:import|from)\\s+${name}\\b`).test(src);

let tifffileReady = false;

// scikit-image can only read TIFFs through tifffile, which is pure Python and so is
// NOT in the Pyodide lockfile — loadPackagesFromImports will never fetch it. TIFF is
// the microscopy format, so `io.imread("thing.tif")` failing on a fresh kernel is the
// single most likely first thing to go wrong. Install it on demand the first time a
// cell touches skimage or tifffile, which keeps it off the boot path.
async function ensureTifffile(src) {
  if (tifffileReady) return;
  if (!imports(src, "skimage") && !imports(src, "tifffile")) return;
  await pyodide.loadPackage("micropip");
  await pyodide.runPythonAsync("import micropip; await micropip.install('tifffile')");
  tifffileReady = true;
}

async function ensurePackages(src) {
  // Pyodide scans the source for import statements and fetches only what it names;
  // already-loaded packages are a no-op on repeat calls.
  await pyodide.loadPackagesFromImports(src);
  // Importing a shim pulls in _ptero_bridge, which needs numpy — and a cell can import
  // a shim without importing numpy itself, which the scan above would never catch.
  if (SHIM_ROOTS.some(name => imports(src, name))) await pyodide.loadPackage("numpy");
  await ensureTifffile(src);
}

async function ensureMatplotlib() {
  if (mplReady) return;
  // Only configure once matplotlib is genuinely present — running MPL_SETUP earlier
  // would itself drag in the package we are trying not to load eagerly.
  const loaded = pyodide.runPython("'matplotlib' in __import__('sys').modules or " +
                                   "__import__('importlib.util', fromlist=['x']).find_spec('matplotlib') is not None");
  if (!loaded) return;
  pyodide.runPython(MPL_SETUP);
  mplReady = true;
}

// ---- running ----------------------------------------------------------------------
// A call site the auto-await rewrite wraps adds one extra frame to any traceback raised
// from inside it — cell output shows the full, untruncated traceback (see cells.js), so
// that implementation detail would otherwise leak straight to the user. Strip it here,
// line-based rather than one big regex, since a traceback's "File ..." line and its
// source line are two separate lines, not a fixed-width block.
function stripAutoAwaitFrames(msg) {
  const lines = msg.split("\n");
  const out = [];
  for (let i = 0; i < lines.length; i++) {
    if (/^\s*File ".*", line \d+, in _ptero_auto_await\s*$/.test(lines[i])) {
      i++;   // also drop the source line pyodide prints under the frame, if present
      continue;
    }
    out.push(lines[i]);
  }
  return out.join("\n");
}

/**
 * Run a chunk of Python and return a display bundle.
 *
 * Every PyProxy created along the way is consumed and destroyed here, so callers deal
 * only in strings — no proxy lifetimes leak into the UI layer.
 *
 * Returns { text, html, figs } — figs is an array of base64 PNGs.
 * Throws on Python error (message is the Python traceback's last line).
 */
export async function run(src, { onOutput } = {}) {
  if (!pyodide) throw new Error("kernel not booted");
  const buffer = [];
  const prev = onStdout;
  onStdout = (m) => { buffer.push(m); prev(m); onOutput?.(m); };
  try {
    await ensurePackages(src);
    await ensureMatplotlib();

    // Invisible sync/await: rewrites `src` so every call is awaited-if-needed, so
    // unmodified upstream-style code (no `await`) works whether or not the browser has
    // JS Promise Integration. See shims/_ptero_autoawait.py.
    const transform = pyodide.globals.get("_ptero_auto_await_source");
    const transformed = transform(src);
    transform.destroy?.();

    let result;
    try {
      result = await pyodide.runPythonAsync(transformed);
    } catch (e) {
      e.message = stripAutoAwaitFrames(e.message ?? String(e));
      throw e;
    }

    let html = null, resultText = null;
    if (result !== undefined && result !== null) {
      // Any object with _repr_html_ (every pandas DataFrame, for one) renders as a
      // real table. Accessing a missing attribute on a PyProxy is safe and returns
      // undefined, but a _repr_html_ that raises is not — hence the try.
      try { html = result._repr_html_ ? result._repr_html_() : null; } catch { html = null; }
      if (html == null) resultText = result?.toString ? result.toString() : String(result);
      result?.destroy?.();
    }

    let figs = [];
    if (mplReady) {
      const popper = pyodide.globals.get("_pop_mpl_figures");
      if (popper) {
        const proxy = popper();
        figs = proxy.toJs();
        proxy.destroy?.();
        popper.destroy?.();
      }
    }

    const text = [buffer.join("\n"), resultText].filter(Boolean).join("\n");
    return { text: text || null, html, figs };
  } finally {
    onStdout = prev;
  }
}

/**
 * Names, types and shapes of everything in the shared cell namespace, plus the
 * workspace listing. This is what lets the agent build mocks that match reality
 * instead of guessing (see agent/tools.js `inspect_user_kernel`).
 */
export function describeGlobals() {
  if (!pyodide) return { vars: [], files: [] };
  const json = pyodide.runPython(String.raw`
def _ptero_describe():
    import json, os
    out = []
    for name, val in list(globals().items()):
        if name.startswith('_') or isinstance(val, type(json)):
            continue
        entry = {'name': name, 'type': type(val).__name__}
        for attr in ('shape', 'dtype', 'columns'):
            v = getattr(val, attr, None)
            if v is not None:
                entry[attr] = str(list(v)) if attr == 'columns' else str(v)
        if not hasattr(val, 'shape') and hasattr(val, '__len__'):
            try: entry['len'] = len(val)
            except Exception: pass
        out.append(entry)
    try: files = sorted(f for f in os.listdir(os.getcwd()) if not f.startswith('.'))
    except OSError: files = []
    return json.dumps({'vars': out, 'files': files})
_ptero_describe()
`);
  return JSON.parse(json);
}

// ---- workspace --------------------------------------------------------------------
let nativefs = null;

// Emscripten's syncfs is bidirectional and treats "local doesn't know about this
// entry" as "delete it" on whichever side is the destination. `nativefs.syncfs()`
// (what mountNativeFS() hands back) only ever runs it as a *push* — pyodide's mem FS
// as source of truth, the real folder as destination — so a file dropped into the
// folder from outside the tab (Finder, another process, a second window), which
// pyodide's mem FS has no node for, reads as "shouldn't exist" and gets deleted from
// the real folder. Confirmed by direct repro, not just inferred from the source.
//
// A *pull* first — populate=true, real folder as source — closes that gap: it can
// only ever add or remove nodes in pyodide's own mem FS mirror, never touch the real
// folder, so it's safe to run unconditionally. Once local knows about every file the
// pull found, a push after it can no longer mistake any of them for garbage. Both
// directions go through the raw global `pyodide.FS.syncfs`, not the `nativefs`
// object's own one-directional method, because that's the only way to request a pull
// at all — and it also means picking up externally-added files no longer needs the
// unmount+remount dance a previous fix here relied on (mountNativeFS's own populate
// pull was never the part that was missing; a *safe* push was).
function fsSyncfs(populate) {
  return new Promise((resolve, reject) => {
    pyodide.FS.syncfs(populate, (err) => (err ? reject(err) : resolve()));
  });
}

async function syncBothWays() {
  await fsSyncfs(true);
  await fsSyncfs(false);
}

export async function mountWorkspace(handle) {
  if (!pyodide) throw new Error("kernel not booted");
  if (nativefs) {
    await syncBothWays();
    pyodide.FS.unmount("/workspace");
  } else {
    try { pyodide.FS.mkdir("/workspace"); } catch { /* exists */ }
  }
  nativefs = await pyodide.mountNativeFS("/workspace", handle);
  pyodide.runPython("import os; os.chdir('/workspace')");
}

export function hasWorkspace() { return nativefs !== null; }

/** Flush pending writes to disk and list what's there now — pulling first (see
 * syncBothWays above) so files added from outside the tab both show up *and*, more
 * importantly, are never mistaken for something to delete. Called after every cell
 * run as well as on an explicit refresh; there's no cheaper partial sync that's still
 * safe, since any push needs the preceding pull to know what not to delete. */
export async function syncWorkspace() {
  if (!pyodide) return [];
  if (nativefs) await syncBothWays();
  const dir = nativefs ? "/workspace" : pyodide.runPython("__import__('os').getcwd()");
  try {
    return pyodide.FS.readdir(dir).filter(n => n !== "." && n !== "..").sort();
  } catch {
    return [];
  }
}

/** Read a file out of the Pyodide FS as bytes, for download links. */
export function readFile(name) {
  return pyodide.FS.readFile(name);
}
