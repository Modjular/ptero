// The agent's hidden scratch kernel: a second Pyodide, in a Worker, with mocked
// segmentation.
//
// This is the "write ≠ run" half of the harness. The agent drafts code, runs it here,
// and only code that survives goes anywhere near the user's notebook. Being a Worker
// matters: a syntax error, an infinite loop or a 400 MB allocation in a draft must not
// freeze the notebook the person is actually using.
//
// It loads no weights and touches no GPU — see shims/_ptero_mock.py for why testing
// against synthetic label maps is the right trade here. The workspace folder's real
// files ARE available at /workspace, read-only-by-construction (see mountWorkspace
// below) — a draft that loads the user's actual image is exactly what we want to catch
// shape/dtype mistakes before push_to_ui, it's only segmentation itself that stays fake.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.28.0/full/pyodide.js");

const SHIM_FILES = [
  "_ptero_bridge.py", "_ptero_autoawait.py", "_ptero_mock.py", "ptero.py",
  "cellpose/__init__.py", "cellpose/models.py",
  "stardist/__init__.py", "stardist/models.py",
  "csbdeep/__init__.py", "csbdeep/utils.py",
  "instanseg/__init__.py",
];
const SHIM_DIR = "/lib/ptero_shims";
const SHIM_ROOTS = ["cellpose", "stardist", "instanseg", "csbdeep", "ptero"];

let py = null;
let out = [];
let mockInstalled = false;
let tifffileReady = false;
let workspaceHandle = null;
let workspaceReady = false;

const imports = (src, name) => new RegExp(`\\b(?:import|from)\\s+${name}\\b`).test(src);

// A call site the auto-await rewrite wraps adds one extra frame to any traceback raised
// from inside it. `error` below is already truncated to the last 6 lines for the agent's
// context budget, so this has to run *before* that slice, or the extra frame could push
// a genuinely useful line out of the window.
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

async function boot(shimBase, catalogueJson) {
  py = await loadPyodide({
    indexURL: "https://cdn.jsdelivr.net/pyodide/v0.28.0/full/",
    stdout: (m) => out.push(m),
    stderr: (m) => out.push(m),
  });

  const sources = await Promise.all(
    SHIM_FILES.map(f => fetch(shimBase + f).then(r => r.text())));
  const mkdir = (p) => { try { py.FS.mkdir(p); } catch { /* exists */ } };
  mkdir("/lib"); mkdir(SHIM_DIR);
  SHIM_FILES.forEach((f, i) => {
    const slash = f.lastIndexOf("/");
    if (slash > -1) mkdir(`${SHIM_DIR}/${f.slice(0, slash)}`);
    py.FS.writeFile(`${SHIM_DIR}/${f}`, sources[i]);
  });
  py.runPython(`import sys; sys.path.insert(0, ${JSON.stringify(SHIM_DIR)})`);
  // Unconditional, same reasoning as kernel.js: draft code never writes `await` under
  // the new design, and this Worker runs in the same browser as the main kernel, so it
  // needs the same invisible auto-await rewrite to actually validate GPU-shaped code on
  // a browser without JS Promise Integration.
  py.runPython("from _ptero_autoawait import _ptero_auto_await_source, _ptero_auto_await");

  // ptero.models.list() must answer here too — the agent asks the scratch kernel what
  // models exist. There is no registry in this Worker, so serve the parent's snapshot.
  globalThis.pteroCatalogue = () => catalogueJson;
  globalThis.pteroEnvironment = async () =>
    JSON.stringify({ gpu: "mocked (scratch kernel)", models: [], resident: [] });
}

// Real files, copied in, never mounted. kernel.js mounts the actual directory handle
// with readwrite permission for the notebook kernel; that permission grant is shared by
// anything holding an equivalent handle to the same directory, so a native mount here
// would inherit write access too, no matter what mode we requested. Copying bytes once
// into this worker's own in-memory FS sidesteps that entirely — draft code can delete,
// truncate or overwrite its `/workspace` freely and the real folder never sees it,
// because after this copy there is no reference to the real handle left in Python's
// reach at all. Additive-only, same reasoning as kernel.js's pullNewFiles: a name this
// FS already has is left alone, so a test run can't undo a copy a previous one made.
async function pullWorkspaceFiles() {
  if (!workspaceHandle) return;
  for await (const [name, entry] of workspaceHandle.entries()) {
    if (entry.kind !== "file") continue;
    try { py.FS.stat(`/workspace/${name}`); continue; } catch { /* not seen yet */ }
    const buf = new Uint8Array(await (await entry.getFile()).arrayBuffer());
    py.FS.writeFile(`/workspace/${name}`, buf);
  }
}

// Called on every test() that has a handle, so a folder chosen — or changed, or added
// to — mid-conversation shows up without the agent needing to know that happened.
// pullWorkspaceFiles is cheap to call repeatedly: an unchanged folder costs one
// directory listing and a stat per name, no bytes moved.
async function mountWorkspace(handle) {
  if (!workspaceReady) {
    try { py.FS.mkdir("/workspace"); } catch { /* exists */ }
    py.runPython("import os; os.chdir('/workspace')");
    workspaceReady = true;
  }
  workspaceHandle = handle;
  await pullWorkspaceFiles();
}

async function ensureDeps(src) {
  await py.loadPackagesFromImports(src);
  const usesShim = SHIM_ROOTS.some(name => imports(src, name));
  if (usesShim) await py.loadPackage("numpy");
  if (usesShim && !mockInstalled) {
    await py.runPythonAsync("import _ptero_mock");
    mockInstalled = true;
  }
  if (!tifffileReady && (imports(src, "skimage") || imports(src, "tifffile"))) {
    await py.loadPackage("micropip");
    await py.runPythonAsync("import micropip; await micropip.install('tifffile')");
    tifffileReady = true;
  }
  if (imports(src, "matplotlib")) {
    // Headless: force Agg and make plt.show() a no-op, or draft plotting code fails
    // here for reasons that have nothing to do with the draft.
    await py.runPythonAsync(
      "import matplotlib; matplotlib.use('Agg')\n" +
      "import matplotlib.pyplot as _p; _p.show = lambda *a, **k: None; del _p");
  }
}

// Build the mock variables the draft expects to find, then run it.
async function test({ code, vars = [], workspaceHandle: handle }) {
  out = [];
  if (handle) await mountWorkspace(handle);
  const setup = [];
  if (vars.length) {
    await py.loadPackage("numpy");
    if (!mockInstalled) { await py.runPythonAsync("import _ptero_mock"); mockInstalled = true; }
    setup.push("import numpy as _np, _ptero_mock as _m");
    vars.forEach((v, i) => {
      const shape = JSON.stringify(v.shape ?? [256, 256]);
      setup.push(`${v.name} = _m.sample_image(${shape}, ${JSON.stringify(v.kind ?? "image")}, ${i})`);
      if (v.dtype) setup.push(`${v.name} = ${v.name}.astype(_np.${v.dtype})`);
    });
  }
  if (setup.length) await py.runPythonAsync(setup.join("\n"));

  try {
    await ensureDeps(code);
    // Invisible sync/await: same rewrite as the main kernel (see kernel.js and
    // shims/_ptero_autoawait.py) — draft code never writes `await`, so this Worker,
    // which runs in the same browser, needs the same transform to actually exercise
    // GPU-shaped code correctly on a browser without JS Promise Integration.
    const transformed = py.globals.get("_ptero_auto_await_source")(code);
    const result = await py.runPythonAsync(transformed);
    let repr = null;
    if (result !== undefined && result !== null) {
      repr = String(result?.toString ? result.toString() : result).slice(0, 2000);
      result?.destroy?.();
    }
    return { ok: true, stdout: out.join("\n").slice(0, 4000), result: repr, vars: describe() };
  } catch (e) {
    return {
      ok: false,
      stdout: out.join("\n").slice(0, 2000),
      // Only the last few traceback lines: the Pyodide frames in between are noise
      // that would eat the agent's context without telling it anything. Strip the
      // auto-await wrapper frame first, or it could push a useful line out of the
      // window.
      error: stripAutoAwaitFrames(String(e.message || e).trim()).split("\n").slice(-6).join("\n"),
      vars: describe(),
    };
  }
}

function describe() {
  try {
    return JSON.parse(py.runPython(String.raw`
def _d():
    import json
    o = []
    for k, v in list(globals().items()):
        if k.startswith('_') or type(v).__name__ in ('module', 'function', 'type'):
            continue
        e = {'name': k, 'type': type(v).__name__}
        for a in ('shape', 'dtype'):
            x = getattr(v, a, None)
            if x is not None: e[a] = str(x)
        cols = getattr(v, 'columns', None)
        if cols is not None: e['columns'] = list(map(str, cols))
        if 'shape' not in e and hasattr(v, '__len__'):
            try: e['len'] = len(v)
            except Exception: pass
        o.append(e)
    return json.dumps(o)
_d()
`));
  } catch {
    return [];
  }
}

// Draft code runs in the shared namespace so a multi-step test can build on itself,
// but a fresh start has to be possible when the agent changes approach.
function reset() {
  py.runPython("for _k in [k for k in list(globals()) if not k.startswith('_')]: del globals()[_k]");
}

self.onmessage = async (e) => {
  const { id, cmd, payload } = e.data;
  try {
    let result;
    if (cmd === "boot") result = await boot(payload.shimBase, payload.catalogue);
    else if (cmd === "test") result = await test(payload);
    else if (cmd === "reset") result = reset();
    else throw new Error(`unknown command ${cmd}`);
    self.postMessage({ id, ok: true, result });
  } catch (err) {
    self.postMessage({ id, ok: false, error: String(err.message || err) });
  }
};
