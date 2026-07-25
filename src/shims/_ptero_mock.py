"""Mock segmentation for the agent's scratch kernel. Never loaded in the user's kernel.

The agent tests its draft code before putting it in front of anyone. What that testing
needs to catch is shape mismatches, wrong tuple arity, misspelled regionprops
properties, bad merges — bugs in the *analysis* code. It does not need real inference:
running the actual network would cost a GPU pass per attempt and, worse, would tempt
the agent to tune thresholds against an image it hasn't been asked about.

So this replaces the bridge with synthetic label maps. The critical property is that
they are *plausible enough that downstream code really runs*: consecutive labels from
1, sensible object sizes, and enough objects that a groupby or a merge has something to
chew on. An empty or single-object mask would let broken code pass by accident.

Patching `_ptero_bridge.segment` covers all three shims at once, because each of them
calls it through the module (`_b.segment(...)`) rather than binding it at import.
"""

import numpy as np

import _ptero_bridge as _b

# Roughly how big an object each model tends to produce, used to pick a plausible
# object count for a given image size. Nuclei models make many small ones; cyto3 makes
# fewer, larger ones (and scales with the diameter it was given).
_TYPICAL_DIAMETER = {
    "cellpose-cyto3": 60.0,
    "stardist-fluo": 22.0,
    "stardist-he": 20.0,
    "instanseg-brightfield": 24.0,
}


def _synthetic_labels(h, w, diameter):
    """A grid of jittered ellipses, labelled 1..n. Deterministic for a given size."""
    rng = np.random.default_rng(abs(hash((h, w, round(diameter)))) % (2**32))
    radius = max(3.0, diameter / 2.0)
    step = max(int(radius * 2.6), 6)
    labels = np.zeros((h, w), dtype=np.int32)
    yy, xx = np.mgrid[0:h, 0:w]

    n = 0
    for cy in range(step, h - step + 1, step):
        for cx in range(step, w - step + 1, step):
            n += 1
            oy, ox = rng.integers(-step // 6, step // 6 + 1, size=2)
            ry = radius * rng.uniform(0.75, 1.15)
            rx = radius * rng.uniform(0.75, 1.15)
            blob = ((yy - cy - oy) / ry) ** 2 + ((xx - cx - ox) / rx) ** 2 <= 1.0
            labels[blob] = n

    if n == 0:
        # Image smaller than one object: still return something segmentable, or the
        # agent gets an empty mask and concludes its code is broken when it isn't.
        labels[h // 4:max(h // 4 + 1, 3 * h // 4), w // 4:max(w // 4 + 1, 3 * w // 4)] = 1

    # Overlap during rasterisation can leave a label fully painted over. Relabel so the
    # ids really are consecutive — code doing `range(1, labels.max()+1)` must not hit
    # a missing id, which is exactly the kind of bug this mock exists to surface.
    present = np.unique(labels)
    present = present[present > 0]
    remap = np.zeros(int(labels.max()) + 1, dtype=np.int32)
    remap[present] = np.arange(1, len(present) + 1, dtype=np.int32)
    return remap[labels]


async def _mock_segment(model_id, img, channels=None, **opts):
    planes = _b._as_planes(img, channels)
    h, w = planes[0].shape
    diameter = opts.get("diameter") or _TYPICAL_DIAMETER.get(model_id, 24.0)
    return _synthetic_labels(h, w, float(diameter))


_b.segment = _mock_segment
MOCKED = True


def sample_image(shape, kind="image", seed=0):
    """A stand-in input array with structure, not noise.

    Flat arrays make normalisation and thresholding behave unlike anything real, so
    these carry a background gradient, bright blobs and a little noise — enough that
    percentile normalisation, intensity measurement and histograms all produce
    something sane.
    """
    rng = np.random.default_rng(seed)
    shape = tuple(int(s) for s in shape)
    if kind == "labels":
        h, w = shape[:2]
        return _synthetic_labels(h, w, 24.0)

    h, w = shape[0], shape[1]
    yy, xx = np.mgrid[0:h, 0:w]
    base = 0.15 + 0.1 * (yy / max(h - 1, 1)) + 0.05 * (xx / max(w - 1, 1))
    blobs = np.zeros((h, w), dtype=np.float32)
    for _ in range(max(4, (h * w) // 6000)):
        cy, cx = rng.integers(0, h), rng.integers(0, w)
        r = rng.uniform(4, max(5.0, min(h, w) / 12))
        blobs += np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * r * r)))
    img = (base + blobs + rng.normal(0, 0.01, (h, w))).astype(np.float32)

    if len(shape) == 2:
        return img
    # Channel axis: give each channel a slightly different structure, so code that
    # picks the wrong channel produces visibly different numbers rather than the same.
    chans = [img * (0.6 + 0.4 * i) for i in range(shape[2] if len(shape) == 3 else 1)]
    return np.stack(chans, axis=-1).astype(np.float32)
