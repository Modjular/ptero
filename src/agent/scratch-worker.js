// The agent's hidden scratch kernel: a second Pyodide, in a Worker, with mocked
// segmentation.
//
// This is the "write ≠ run" half of the harness. The agent drafts code, runs it here,
// and only code that survives goes anywhere near the user's notebook. Being a Worker
// matters: a syntax error, an infinite loop or a 400 MB allocation in a draft must not
// freeze the notebook the person is actually using.
//
// It loads no weights and touches no GPU — see shims/_ptero_mock.py for why testing
// against synthetic label maps is the right trade here.
importScripts("https://cdn.jsdelivr.net/pyodide/v0.28.0/full/pyodide.js");

const SHIM_FILES = [
  "_ptero_bridge.py", "_ptero_mock.py", "ptero.py",
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

const imports = (src, name) => new RegExp(`\\b(?:import|from)\\s+${name}\\b`).test(src);

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

  // ptero.models.list() must answer here too — the agent asks the scratch kernel what
  // models exist. There is no registry in this Worker, so serve the parent's snapshot.
  globalThis.pteroCatalogue = () => catalogueJson;
  globalThis.pteroEnvironment = async () =>
    JSON.stringify({ gpu: "mocked (scratch kernel)", models: [], resident: [] });
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
async function test({ code, vars = [] }) {
  out = [];
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
    const result = await py.runPythonAsync(code);
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
      // that would eat the agent's context without telling it anything.
      error: String(e.message || e).trim().split("\n").slice(-6).join("\n"),
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
