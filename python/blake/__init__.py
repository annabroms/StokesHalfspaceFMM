"""Blake velocity; image representation by Gimbutas, Greengard and Veerapaneni."""

from .geometry import FMMResolutionError, Geometry, SingularInteractionError, prepare
from .direct import blake_pair, direct_velocity
from .evaluate import apply, velocity
from .diagnostics import constituent_scale, metrics

__all__ = [
    "velocity",
    "apply",
    "prepare",
    "Geometry",
    "blake_pair",
    "direct_velocity",
    "constituent_scale",
    "metrics",
    "SingularInteractionError",
    "FMMResolutionError",
]
