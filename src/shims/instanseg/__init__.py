"""A stand-in for the `instanseg` package, backed by the WebGPU InstanSeg engine.

    from instanseg import InstanSeg
    model = InstanSeg('brightfield_nuclei')
    labels, image = model.eval_small_image(img)

Deviations from upstream — read these before copying upstream code verbatim:
  * Upstream returns **torch tensors** (shaped [1,1,H,W]) and expects torch to be
    available. This build has no torch: `eval_small_image` returns a plain 2-D int32
    **numpy** array, which is what the rest of the scientific stack here wants anyway.
    Upstream code doing `labels.cpu().numpy().squeeze()` will fail — drop that call.
  * Only the `brightfield_nuclei` checkpoint is available.
  * `pixel_size` is accepted and ignored — there is no resolution-adaptive rescaling in
    this port, so segment at the image's native resolution.
"""

import _ptero_bridge as _b

__all__ = ["InstanSeg"]
__version__ = "0.0.8+ptero"

_AVAILABLE = {"brightfield_nuclei": "instanseg-brightfield"}


class InstanSeg:
    def __init__(self, model_type="brightfield_nuclei", device=None, image_reader=None,
                 verbosity=1, **kwargs):
        model_id = _AVAILABLE.get(model_type)
        if model_id is None:
            raise ValueError(
                f"unknown InstanSeg model {model_type!r}. Available in the browser: "
                + ", ".join(sorted(_AVAILABLE))
            )
        self.model_type = model_type
        self._model_id = model_id

    @_b.maybe_sync
    async def eval_small_image(self, image, pixel_size=None, target="nuclei",
                               seed_threshold=None, mask_threshold=None, **kwargs):
        """Segment one image that fits in memory. Returns (labels, image).

        `labels` is a 2-D int32 numpy array (0 = background), NOT a torch tensor.
        """
        labels = await _b.segment(self._model_id, image,
                                  seed_threshold=seed_threshold,
                                  mask_threshold=mask_threshold,
                                  min_size=kwargs.get("min_size"))
        return labels, image

    # Upstream's whole-slide entry points. Present so code reaching for them gets a
    # useful message instead of AttributeError.
    def eval_medium_image(self, *args, **kwargs):
        raise NotImplementedError(
            "eval_medium_image (tiled whole-slide inference) is not part of the browser "
            "build — use eval_small_image, which handles the whole image at once."
        )

    eval_whole_slide_image = eval_medium_image

    def __repr__(self):
        return f"InstanSeg({self.model_type!r})"
