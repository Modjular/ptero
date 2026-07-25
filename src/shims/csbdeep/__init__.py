"""A minimal stand-in for `csbdeep`, present because StarDist's examples import
`normalize` from it. Only `csbdeep.utils.normalize` exists here."""

from . import utils

__all__ = ["utils"]
