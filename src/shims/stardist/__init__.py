"""A stand-in for the `stardist` package, backed by the WebGPU StarDist engine.

See `stardist.models` for the API and the list of deviations from upstream. Note that
`csbdeep.utils.normalize` — which nearly every upstream StarDist example calls before
predicting — is also provided in this kernel.
"""

from . import models

__all__ = ["models"]
__version__ = "0.9.1+ptero"
