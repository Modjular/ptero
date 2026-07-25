"""A stand-in for the `cellpose` package, backed by the WebGPU cyto3 engine.

See `cellpose.models` for the API and the list of deviations from upstream.
"""

from . import models

__all__ = ["models"]
__version__ = "3.1.1+ptero"
