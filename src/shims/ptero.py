"""`ptero` — what this browser kernel can actually do.

The three segmentation shims deliberately imitate their upstream libraries, which means
none of them can answer "which models are available here, and which should I use?".
That question is what this module is for. It is the discovery surface: an agent (or a
person) can call `ptero.models.list()` and find out what is installed rather than
guessing from memory.

    import ptero
    ptero.models.list()          # every model, what it's for, its default parameters
    ptero.models.describe(id)    # one model
    ptero.models.suggest(...)    # id for a described imaging situation
    ptero.workspace.files()      # what's in the mounted folder
    ptero.env                    # GPU, sync-vs-await calling convention, loaded weights
"""

import os

import _ptero_bridge as _b


class _Models:
    def list(self):
        """Every available model as a list of dicts (id, good_for, defaults, ...)."""
        return _b.catalogue()

    def describe(self, model_id):
        for m in self.list():
            if m["id"] == model_id:
                return m
        raise KeyError(f"unknown model {model_id!r} — try ptero.models.list()")

    def ids(self):
        return [m["id"] for m in self.list()]

    def suggest(self, stain=None, target="nuclei", modality=None):
        """Rough routing from an imaging situation to a model id.

        Deliberately a small set of rules rather than anything clever — it encodes the
        same table the docs give, so the answer is inspectable and arguable. When in
        doubt about the biology, ask the person whose image it is instead of guessing.

            stain     e.g. 'dapi', 'hoechst', 'h&e', 'none'
            target    'nuclei' or 'cells'/'cytoplasm'
            modality  e.g. 'fluorescence', 'brightfield', 'phase'
        """
        s = (stain or "").lower()
        m = (modality or "").lower()
        t = (target or "").lower()

        if "h&e" in s or "he" == s or "eosin" in s or "histology" in m:
            return "stardist-he"
        if t.startswith("cell") or "cyto" in t:
            return "cellpose-cyto3"
        if any(k in s for k in ("dapi", "hoechst", "draq", "sytox")) or "fluor" in m:
            return "stardist-fluo"
        if any(k in m for k in ("brightfield", "bright-field", "phase", "dic", "transmitted")):
            return "instanseg-brightfield"
        # Nothing matched confidently: cyto3 is the generalist, and the caller should
        # be told the choice was a fallback rather than a diagnosis.
        return "cellpose-cyto3"

    def __repr__(self):
        return "\n".join(f"{m['id']:<24} {m['good_for']}" for m in self.list())


class _Workspace:
    """The folder the notebook has mounted, if any (otherwise the in-memory FS)."""

    @property
    def path(self):
        return "/workspace" if os.path.isdir("/workspace") else os.getcwd()

    def files(self):
        try:
            return sorted(f for f in os.listdir(self.path) if not f.startswith("."))
        except OSError:
            return []

    def __repr__(self):
        names = self.files()
        return f"{self.path}: " + (", ".join(names) if names else "(empty)")


class _Env:
    """Runtime facts worth knowing before writing code against this kernel.

    `sync_calls` is the one that changes how you write code: True means the
    segmentation shims are ordinary synchronous functions, False means they are
    coroutines and need `await`. It is read live rather than cached, because the
    answer depends on how the current code was invoked — see _ptero_bridge.
    """

    backend = "WebGPU (hand-written WGSL, no ML framework)"

    @property
    def sync_calls(self):
        return _b.sync_calls()

    def __getitem__(self, key):
        return getattr(self, key)

    def asdict(self):
        return {"sync_calls": self.sync_calls, "backend": self.backend}

    def __repr__(self):
        return repr(self.asdict())


models = _Models()
workspace = _Workspace()
env = _Env()


@_b.maybe_sync
async def gpu():
    """GPU adapter and which weight sets are currently resident."""
    import json

    import js
    return json.loads(await js.pteroEnvironment())
