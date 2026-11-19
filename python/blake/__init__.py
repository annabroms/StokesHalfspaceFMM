"""Blake velocity, pressure, gradients and traction.

The half-space image representation is by Gimbutas, Greengard and Veerapaneni.
"""

from .geometry import FMMResolutionError, Geometry, SingularInteractionError, prepare
from .direct import blake_pair, direct_fields, direct_velocity
from .evaluate import apply, apply_fields, fields, velocity
from .diagnostics import constituent_scale, metrics

__all__ = [
    "velocity",
    "apply",
    "fields",
    "apply_fields",
    "prepare",
    "Geometry",
    "blake_pair",
    "direct_velocity",
    "direct_fields",
    "constituent_scale",
    "metrics",
    "SingularInteractionError",
    "FMMResolutionError",
]
