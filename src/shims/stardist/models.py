"""`stardist.models`, backed by the WebGPU StarDist engine.

Two pretrained checkpoints are available, under their upstream names:

    StarDist2D.from_pretrained('2D_versatile_fluo')   fluorescent nuclei (DAPI/Hoechst)
    StarDist2D.from_pretrained('2D_versatile_he')     H&E histology nuclei (RGB)

Deviations from upstream:
  * `predict_instances` returns `(labels, details)` where `details` carries only
    `{'n': <object count>}`. Upstream also returns per-object polygon coordinates,
    probabilities and centre points; the WGSL decode rasterises polygons on the GPU and
    does not surface them. For per-object geometry use `skimage.measure.regionprops`
    on the returned label map.
  * There is no `predict_instances_big` / tiling API — the engine handles the whole
    image directly.
"""

import _ptero_bridge as _b

__all__ = ["StarDist2D"]

# Upstream pretrained name -> registry model id.
_PRETRAINED = {
    "2D_versatile_fluo": "stardist-fluo",
    "2D_versatile_he": "stardist-he",
}


class StarDist2D:
    def __init__(self, config=None, name=None, basedir=None):
        model_id = _PRETRAINED.get(name)
        if model_id is None:
            raise ValueError(
                f"unknown StarDist model {name!r}. Available in the browser: "
                + ", ".join(sorted(_PRETRAINED))
            )
        self.name = name
        self._model_id = model_id

    @classmethod
    def from_pretrained(cls, name=None):
        if name is None:
            # Upstream prints the list and returns None; being explicit is friendlier.
            raise ValueError("pick a pretrained model: " + ", ".join(sorted(_PRETRAINED)))
        return cls(name=name)

    @classmethod
    def from_pretrained_names(cls):
        return sorted(_PRETRAINED)

    @_b.maybe_sync
    async def predict_instances(self, img, prob_thresh=None, nms_thresh=None,
                                axes=None, scale=None, **kwargs):
        labels = await _b.segment(self._model_id, img,
                                  prob_thresh=prob_thresh, nms_thresh=nms_thresh)
        return labels, {"n": int(labels.max())}

    def __repr__(self):
        return f"StarDist2D({self.name!r})"
