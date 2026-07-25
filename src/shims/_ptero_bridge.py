"""Shared plumbing for the segmentation shims. Not a user-facing module.

Everything the three fake libraries have in common lives here: the marshalling of a
numpy image across to the WebGPU registry, and the decorator that decides whether the
public shim methods are ordinary synchronous functions or coroutines.

Why the decorator exists
------------------------
The GPU boundary is genuinely async, so every shim method is written `async def`. But
the whole point of these shims is that code written against the real cellpose/stardist/
instanseg APIs runs unmodified, and the real APIs are synchronous — a stray `await`
requirement is a per-call failure mode for anyone (or anything) writing from the
upstream docs.

Pyodide's `run_sync` bridges that gap using JS Promise Integration: it blocks a Python
stack on an awaitable, so an async call can be spelled synchronously. So it is detected,
not assumed: where it works the shims are faithfully synchronous, and where it doesn't
they stay coroutines and callers add `await`. `ptero.env.sync_calls` reports which mode
is live.

The detection has to happen per call, not once at import. `can_run_sync()` describes
the *current Python stack*, not the browser: it is True only on a stack entered through
`runPythonAsync` (how notebook cells run) and False on a plain `runPython` stack, in the
very same page. Caching the answer at import time therefore bakes in whichever stack
happened to import the shim first — measured, not theorised. The check itself is cheap.
"""

import functools
import json

import numpy as np
from pyodide.ffi import can_run_sync, run_sync


def sync_calls():
    """True if shim methods are callable without `await` from right here."""
    return bool(can_run_sync())


def maybe_sync(fn):
    """Expose an async shim method synchronously wherever JSPI allows it.

    Falls back to returning the coroutine — so the caller must `await` — on stacks
    where `run_sync` is unavailable.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        coro = fn(*args, **kwargs)
        if can_run_sync():
            return run_sync(coro)
        return coro

    wrapper.__ptero_maybe_sync__ = True
    return wrapper


def _as_planes(img, channels=None):
    """Normalise a user-supplied image into a list of 2-D float32 planes.

    Accepts [H,W], [H,W,C] and [C,H,W]. The [H,W,C] vs [C,H,W] ambiguity is resolved
    the way every imaging tool resolves it — the short axis is the channel axis — with
    a hard cap of 8, because a genuine 2-channel fluorescence stack and a 3-channel RGB
    photo must both work and neither has more than a handful of channels.
    """
    arr = np.asarray(img)
    if arr.ndim == 2:
        planes = [arr]
    elif arr.ndim == 3:
        if arr.shape[0] <= 8 and arr.shape[0] < arr.shape[2]:
            planes = [arr[c] for c in range(arr.shape[0])]          # [C,H,W]
        elif arr.shape[2] <= 8:
            planes = [arr[..., c] for c in range(arr.shape[2])]     # [H,W,C]
        else:
            raise ValueError(
                f"can't tell which axis of a {arr.shape} array is the channel axis — "
                "pass a single 2-D plane, or slice the channel you want first"
            )
    else:
        raise ValueError(f"expected a 2-D or 3-D image, got shape {arr.shape}")

    if channels is not None:
        planes = _apply_channel_spec(planes, channels)
    return [np.ascontiguousarray(p, dtype=np.float32) for p in planes]


def _apply_channel_spec(planes, channels):
    """Cellpose's `channels=[c1, c2]` convention: 0=grayscale, 1=R, 2=G, 3=B.

    c1 is the channel to segment, c2 the optional nuclear channel (0 = none). This is
    real upstream semantics that code copied from the cellpose docs will use, so it has
    to mean what it means there rather than being quietly ignored.
    """
    try:
        c1, c2 = (int(channels[0]), int(channels[1]))
    except (TypeError, IndexError, ValueError):
        raise ValueError("channels must be a pair like [0,0], [2,3] or [1,0]") from None

    def pick(c):
        if c == 0:
            # Grayscale: average the planes, matching cellpose's own behaviour for a
            # multi-channel image with channels=[0,0].
            return planes[0] if len(planes) == 1 else np.mean(planes, axis=0)
        if not 1 <= c <= len(planes):
            raise ValueError(
                f"channels asked for channel {c} but the image has {len(planes)} "
                f"channel(s) (cellpose numbers them 1=R, 2=G, 3=B; 0 = grayscale)"
            )
        return planes[c - 1]

    out = [pick(c1)]
    if c2:
        out.append(pick(c2))
    return out


async def segment(model_id, img, channels=None, **opts):
    """Run a registered model and return an int32 [H,W] label map.

    Planes are stacked into one contiguous [C,H,W] float32 buffer and handed over as a
    single numpy proxy — one `getBuffer` on the JS side, no list-of-proxies marshalling.
    """
    import js  # local: the shared cell namespace must not gain a stray `js` global

    planes = _as_planes(img, channels)
    h, w = planes[0].shape
    for p in planes[1:]:
        if p.shape != (h, w):
            raise ValueError(f"channel planes disagree on size: {(h, w)} vs {p.shape}")

    stack = np.ascontiguousarray(np.stack(planes, axis=0), dtype=np.float32)
    # Options cross as JSON so the JS side sees a plain object rather than a PyProxy
    # dict it would have to convert and destroy.
    clean = {k: v for k, v in opts.items() if v is not None}
    labels_mv = await js.pteroSegment(model_id, stack, len(planes), h, w, json.dumps(clean))
    return np.asarray(labels_mv, dtype=np.int32).reshape(h, w)


def catalogue():
    import js
    return json.loads(js.pteroCatalogue())
