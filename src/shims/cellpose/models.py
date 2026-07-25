"""`cellpose.models`, backed by the WebGPU cyto3 engine.

Mirrors the upstream API closely enough that code copied from the cellpose docs runs
unchanged. Two upstream details that are easy to get wrong, and are honoured here:

  * `Cellpose` and `CellposeModel` are NOT the same class. Upstream, `Cellpose` bundles
    a size model and its `eval` returns a 4-tuple (masks, flows, styles, diams);
    `CellposeModel` has no size model and returns a 3-tuple (masks, flows, styles).
    Aliasing them — as an earlier version of this shim did — silently breaks correct
    upstream unpacking.
  * `channels=[c1, c2]` really selects channels (0=grayscale, 1=R, 2=G, 3=B), with c2
    the optional nuclear channel.

Deviations from upstream, all forced by running on WebGPU in a browser:
  * `flows` and `styles` come back as None. The engine computes flow fields internally
    but does not surface them in cellpose's nested list-of-lists layout.
  * `diams` echoes the diameter you passed — there is no SizeModel port, so `Cellpose`
    cannot estimate a diameter for you. Pass a real one; it is the single most
    important parameter.
  * Only cyto3 weights are available. `model_type` is accepted and recorded, but any
    value maps to cyto3.
"""

import _ptero_bridge as _b

_MODEL_ID = "cellpose-cyto3"

__all__ = ["Cellpose", "CellposeModel"]


class CellposeModel:
    """Upstream `cellpose.models.CellposeModel` — no size model; eval returns 3 values."""

    def __init__(self, gpu=True, model_type="cyto3", pretrained_model=None, device=None, **kwargs):
        self.model_type = model_type or "cyto3"
        self.pretrained_model = pretrained_model
        self.gpu = gpu
        self.diam_mean = 30.0

    @_b.maybe_sync
    async def eval(self, x, diameter=30.0, channels=None, channel_axis=None,
                   flow_threshold=0.4, cellprob_threshold=0.0, min_size=15,
                   niter=None, normalize=True, **kwargs):
        masks = await _b.segment(
            _MODEL_ID, x, channels=channels,
            diameter=float(diameter) if diameter else 30.0,
            flow_threshold=flow_threshold,
            cellprob_threshold=cellprob_threshold,
            min_size=min_size,
            niter=niter,
        )
        return masks, None, None

    def __repr__(self):
        return f"CellposeModel(model_type={self.model_type!r})"


class Cellpose(CellposeModel):
    """Upstream `cellpose.models.Cellpose` — size model + eval returns 4 values.

    The 4th value (`diams`) is the diameter actually used. Upstream that is what the
    SizeModel estimated; here it is simply what you passed, since there is no size
    model in the browser build.
    """

    @_b.maybe_sync
    async def eval(self, x, diameter=30.0, channels=None, **kwargs):
        diam = float(diameter) if diameter else 30.0
        masks = await _b.segment(
            _MODEL_ID, x, channels=channels, diameter=diam,
            flow_threshold=kwargs.get("flow_threshold", 0.4),
            cellprob_threshold=kwargs.get("cellprob_threshold", 0.0),
            min_size=kwargs.get("min_size", 15),
            niter=kwargs.get("niter"),
        )
        return masks, None, None, diam

    def __repr__(self):
        return f"Cellpose(model_type={self.model_type!r})"
