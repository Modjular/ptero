"""The one piece of `csbdeep` that StarDist examples always reach for.

Provided as a real percentile normalisation rather than a no-op, so code that follows
the upstream examples behaves as documented. The engine normalises internally as well,
and that normalisation is idempotent on already-normalised input, so calling this is
harmless either way.
"""

import numpy as np

__all__ = ["normalize"]


def normalize(x, pmin=1, pmax=99.8, axis=None, eps=1e-20, dtype=np.float32, **kwargs):
    """Percentile-normalise to roughly [0, 1].

    Matches upstream defaults (1st and 99.8th percentile). `axis=None` normalises over
    the whole array; pass `axis=(0,1)` for the per-channel behaviour StarDist's RGB
    examples use.
    """
    x = np.asarray(x, dtype=dtype)
    lo = np.percentile(x, pmin, axis=axis, keepdims=True)
    hi = np.percentile(x, pmax, axis=axis, keepdims=True)
    # np.percentile returns float64, so the division above promotes x and the result
    # silently stops honouring the dtype argument. Cast back so a caller that asked
    # for float32 (the default, and what the engines expect) actually gets it.
    return ((x - lo) / np.maximum(hi - lo, eps)).astype(dtype, copy=False)
