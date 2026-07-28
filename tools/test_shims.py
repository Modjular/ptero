"""Exercise the shim logic under plain CPython by faking the pyodide.ffi module.

    python3 tools/test_shims.py           # await mode (no JSPI)
    python3 tools/test_shims.py --sync    # sync mode (JSPI available)

Covers everything in src/shims/ that is pure Python: channel selection, plane
normalisation, the maybe_sync decorator, upstream call signatures and return arities,
and the module structure. The GPU call itself is stubbed, so this needs no browser and
runs in well under a second. Both modes must pass — the shims are callable either way
depending on whether the browser gives us JS Promise Integration.

Only numpy is required.
"""
import asyncio
import sys
import types
from pathlib import Path

import numpy as np

SHIMS = Path(__file__).resolve().parent.parent / "src" / "shims"

# --- fake pyodide.ffi ---------------------------------------------------------
SYNC = "--sync" in sys.argv
pyodide = types.ModuleType("pyodide")
ffi = types.ModuleType("pyodide.ffi")
ffi.can_run_sync = lambda: SYNC


def _fake_run_sync(aw):
    """Fake JSPI's run_sync by manually stepping the coroutine, rather than
    asyncio.run_until_complete: none of the coroutines this file drives (the fake
    pteroSegment below never really awaits a pending Future) actually suspend, and
    unlike run_until_complete, manual stepping works even when called from *inside*
    an already-running event loop. That reentrant case is exactly what the auto-await
    rewrite's own top-level-await coroutine now produces for every cell, regardless of
    SYNC — and real JSPI's stack-switching is built precisely for nested suspension
    like this (Pyodide's own top-level-await isn't even asyncio-driven), so this fake
    needs to tolerate it too, not reject it as asyncio's single flat loop does."""
    try:
        aw.send(None)
    except StopIteration as e:
        return e.value
    raise RuntimeError("fake run_sync: coroutine actually suspended, can't fake it")


ffi.run_sync = _fake_run_sync
pyodide.ffi = ffi
sys.modules["pyodide"] = pyodide
sys.modules["pyodide.ffi"] = ffi

# --- fake `js` global: record the call, return a plausible label map ----------
CALLS = []
js = types.ModuleType("js")


async def pteroSegment(model_id, stack, C, H, W, opts_json):
    import json
    CALLS.append(dict(model=model_id, C=C, H=H, W=W, opts=json.loads(opts_json),
                      stack=np.array(stack)))
    lab = np.zeros((H, W), np.int32)
    lab[2:5, 2:5] = 1
    lab[7:9, 7:9] = 2
    return lab.reshape(-1)


js.pteroSegment = pteroSegment
js.pteroCatalogue = lambda: '[{"id":"cellpose-cyto3"}]'
sys.modules["js"] = js

sys.path.insert(0, str(SHIMS))

ok = fail = 0


def check(name, cond, extra=""):
    global ok, fail
    if cond:
        ok += 1
        print(f"  ok   {name}")
    else:
        fail += 1
        print(f"  FAIL {name} {extra}")


def call(fn, *a, **k):
    """Invoke a shim method in whichever mode the decorator produced."""
    r = fn(*a, **k)
    if asyncio.iscoroutine(r):
        assert not SYNC, "expected a sync call under --sync"
        return asyncio.get_event_loop().run_until_complete(r)
    assert SYNC, "expected a coroutine without --sync"
    return r


print(f"\n=== mode: {'sync (JSPI)' if SYNC else 'async (await)'} ===")

# --- _as_planes / channel spec ------------------------------------------------
import _ptero_bridge as b  # noqa: E402

print("\n_as_planes:")
check("2-D passes through", len(b._as_planes(np.zeros((4, 5)))) == 1)
check("[H,W,C] splits on last axis",
      [p.shape for p in b._as_planes(np.zeros((40, 50, 3)))] == [(40, 50)] * 3)
check("[C,H,W] splits on first axis",
      [p.shape for p in b._as_planes(np.zeros((3, 40, 50)))] == [(40, 50)] * 3)
check("float32 + contiguous", b._as_planes(np.zeros((4, 5), np.float64))[0].dtype == np.float32)
try:
    b._as_planes(np.zeros((40, 50, 30)))
    check("ambiguous 3-D rejected", False)
except ValueError as e:
    check("ambiguous 3-D rejected", "channel axis" in str(e))
try:
    b._as_planes(np.zeros((2, 3, 4, 5)))
    check("4-D rejected", False)
except ValueError:
    check("4-D rejected", True)

print("\nchannels=[c1,c2] (cellpose convention 0=gray 1=R 2=G 3=B):")
img = np.stack([np.full((4, 5), 1.0), np.full((4, 5), 2.0), np.full((4, 5), 3.0)], axis=-1)
p = b._as_planes(img, channels=[2, 3])
check("[2,3] -> G then B", len(p) == 2 and p[0][0, 0] == 2.0 and p[1][0, 0] == 3.0)
p = b._as_planes(img, channels=[1, 0])
check("[1,0] -> R only, no second channel", len(p) == 1 and p[0][0, 0] == 1.0)
p = b._as_planes(img, channels=[0, 0])
check("[0,0] -> mean of all planes", len(p) == 1 and abs(p[0][0, 0] - 2.0) < 1e-6)
try:
    b._as_planes(img, channels=[5, 0])
    check("out-of-range channel rejected", False)
except ValueError as e:
    check("out-of-range channel rejected", "has 3 channel" in str(e))
try:
    b._as_planes(img, channels=7)
    check("malformed channels rejected", False)
except ValueError as e:
    check("malformed channels rejected", "pair like" in str(e))

# --- cellpose -----------------------------------------------------------------
print("\ncellpose:")
from cellpose import models  # noqa: E402

check("Cellpose is not CellposeModel", models.Cellpose is not models.CellposeModel)
CALLS.clear()
r = call(models.CellposeModel(model_type="cyto3").eval, np.zeros((16, 16)), diameter=100)
check("CellposeModel.eval returns 3-tuple", len(r) == 3, f"got {len(r)}")
check("  masks are int32 [H,W]", r[0].shape == (16, 16) and r[0].dtype == np.int32)
check("  masks.max() == 2", r[0].max() == 2)
check("  diameter forwarded", CALLS[-1]["opts"]["diameter"] == 100.0)
check("  model id", CALLS[-1]["model"] == "cellpose-cyto3")

r = call(models.Cellpose(model_type="cyto3").eval, np.zeros((16, 16)), diameter=55)
check("Cellpose.eval returns 4-tuple", len(r) == 4, f"got {len(r)}")
check("  4th value is the diameter used", r[3] == 55.0)

CALLS.clear()
two = np.stack([np.zeros((16, 16)), np.ones((16, 16))], axis=-1)
call(models.CellposeModel().eval, two, diameter=30, channels=[1, 2])
check("two channels reach the bridge", CALLS[-1]["C"] == 2)
check("  stacked [C,H,W] contiguous", CALLS[-1]["stack"].shape == (2, 16, 16))
check("  niter=None dropped from opts", "niter" not in CALLS[-1]["opts"])

# --- stardist -----------------------------------------------------------------
print("\nstardist:")
from stardist.models import StarDist2D  # noqa: E402
from csbdeep.utils import normalize  # noqa: E402

CALLS.clear()
sd = StarDist2D.from_pretrained("2D_versatile_fluo")
labels, details = call(sd.predict_instances, np.zeros((16, 16)))
check("predict_instances -> (labels, details)", labels.shape == (16, 16) and details["n"] == 2)
check("  fluo -> stardist-fluo", CALLS[-1]["model"] == "stardist-fluo")
check("  unset thresholds dropped", CALLS[-1]["opts"] == {})
call(StarDist2D.from_pretrained("2D_versatile_he").predict_instances,
     np.zeros((16, 16, 3)), prob_thresh=0.5)
check("he -> stardist-he, 3 channels", CALLS[-1]["model"] == "stardist-he" and CALLS[-1]["C"] == 3)
check("  prob_thresh forwarded", CALLS[-1]["opts"]["prob_thresh"] == 0.5)
try:
    StarDist2D.from_pretrained("2D_demo")
    check("unknown checkpoint rejected", False)
except ValueError as e:
    check("unknown checkpoint rejected", "2D_versatile_fluo" in str(e))

x = np.array([[0.0, 1, 2, 3, 100]])
n = normalize(x)
check("normalize -> ~[0,1] float32", n.dtype == np.float32 and abs(n.max() - 1.0) < 0.05)

# --- instanseg ----------------------------------------------------------------
print("\ninstanseg:")
from instanseg import InstanSeg  # noqa: E402

CALLS.clear()
lab, im = call(InstanSeg("brightfield_nuclei").eval_small_image, np.zeros((16, 16, 3)))
check("eval_small_image -> (numpy labels, image)",
      isinstance(lab, np.ndarray) and lab.dtype == np.int32 and lab.shape == (16, 16))
check("  model id", CALLS[-1]["model"] == "instanseg-brightfield")
try:
    InstanSeg("fluorescence_nuclei_and_cells")
    check("unknown checkpoint rejected", False)
except ValueError as e:
    check("unknown checkpoint rejected", "brightfield_nuclei" in str(e))
try:
    InstanSeg().eval_medium_image(None)
    check("eval_medium_image explains itself", False)
except NotImplementedError as e:
    check("eval_medium_image explains itself", "eval_small_image" in str(e))

# --- ptero --------------------------------------------------------------------
print("\nptero:")
import ptero  # noqa: E402

check("models.list() from the JS catalogue", ptero.models.list()[0]["id"] == "cellpose-cyto3")
check("env.sync_calls reflects JSPI", ptero.env.sync_calls is SYNC)
check("suggest dapi -> stardist-fluo", ptero.models.suggest(stain="DAPI") == "stardist-fluo")
check("suggest h&e -> stardist-he", ptero.models.suggest(stain="H&E") == "stardist-he")
check("suggest brightfield -> instanseg",
      ptero.models.suggest(modality="brightfield") == "instanseg-brightfield")
check("suggest cells -> cyto3", ptero.models.suggest(target="cells") == "cellpose-cyto3")
check("suggest unknown -> cyto3 fallback", ptero.models.suggest() == "cellpose-cyto3")

# --- _ptero_autoawait (invisible sync/await) -----------------------------------
# Runs the *real* maybe_sync-wrapped cellpose shim through the rewrite, in whichever
# mode this process is running under (SYNC or not) — the whole point of the rewrite is
# that the same unmodified, no-await source produces correct results either way.
print("\n_ptero_autoawait (invisible sync/await):")
import ast as _ast  # noqa: E402

import _ptero_autoawait as aa  # noqa: E402


def run_transformed(src, ns):
    """Execute src after the auto-await rewrite, resolving a top-level coroutine if one
    comes back — mirrors what pyodide.runPythonAsync does for a real cell."""
    ns.setdefault("_ptero_auto_await", aa._ptero_auto_await)
    out = aa._ptero_auto_await_source(src)
    code = compile(out, "<autoawait-test>", "exec", flags=_ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    coro = eval(code, ns)
    if coro is not None:
        asyncio.get_event_loop().run_until_complete(coro)
    return ns


CALLS.clear()
ns = {"model": models.CellposeModel(model_type="cyto3"), "np": np}
run_transformed(
    "masks, flows, styles = model.eval(np.zeros((16, 16)), diameter=100, channels=[0, 0])",
    ns,
)
check("bare call (no await written) resolves correctly", ns["masks"].max() == 2)

ns = {"model": models.CellposeModel(model_type="cyto3"), "np": np}
run_transformed(
    "masks, flows, styles = await model.eval(np.zeros((16, 16)), diameter=100, channels=[0, 0])",
    ns,
)
check("explicit `await` still resolves correctly (no double-await TypeError)",
      ns["masks"].max() == 2)

ns = {"model": models.CellposeModel(model_type="cyto3"), "np": np}
run_transformed(
    "coro = model.eval(np.zeros((16, 16)), diameter=100, channels=[0, 0])\n"
    "masks, flows, styles = await coro",
    ns,
)
check("coroutine stashed then awaited later resolves correctly", ns["masks"].max() == 2)

ns = {"model": models.CellposeModel(model_type="cyto3"), "np": np}
run_transformed(
    "results = [model.eval(np.zeros((16, 16)), diameter=100, channels=[0, 0])[0] "
    "for _ in range(2)]",
    ns,
)
check("call inside a comprehension resolves correctly",
      len(ns["results"]) == 2 and all(r.max() == 2 for r in ns["results"]))

src = ("def helper(img):\n"
       "    masks, _, _ = model.eval(img, diameter=100, channels=[0, 0])\n"
       "    return masks")
out = aa._ptero_auto_await_source(src)
try:
    compile(out, "<t>", "exec", flags=_ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    compiled = True
except SyntaxError:
    compiled = False
check("nested def body left untouched (documented scope limit) and still compiles",
      "await" not in out and compiled, out)

src = "handler = lambda img: model.eval(img, diameter=100, channels=[0, 0])"
out = aa._ptero_auto_await_source(src)
try:
    compile(out, "<t>", "exec", flags=_ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
    compiled = True
except SyntaxError:
    compiled = False
check("lambda body left untouched (await is illegal there) and still compiles",
      "await" not in out and compiled, out)

print(f"\n{ok} passed, {fail} failed")
sys.exit(1 if fail else 0)
