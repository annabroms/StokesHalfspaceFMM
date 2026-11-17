"""Validated, translated geometry and bounded-memory singularity checks."""

from dataclasses import dataclass
from itertools import product

import numpy as np


class SingularInteractionError(ValueError):
    """An exact physical source/target coincidence has no finite velocity."""


class FMMResolutionError(ValueError):
    """The FMM exclusion radius cannot resolve a physical or image interaction."""


def _array(value, name):
    """Validate a finite real component-first array and own its float64 storage."""
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] != 3 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric (3, N) array")
    out = np.array(raw, dtype=np.float64, order="F", copy=True)
    if not np.isfinite(out).all():
        raise ValueError(f"{name} must contain only finite values")
    return out


def _scalar(value, name, positive=False):
    """Validate a finite real scalar without silently discarding imaginary parts."""
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result) or (positive and result <= 0):
        raise ValueError(f"{name} must be finite" + (" and positive" if positive else ""))
    return result


@dataclass(frozen=True)
class Geometry:
    """Opaque prepared geometry; use prepare(), and do not mutate its arrays."""

    sources: np.ndarray
    targets: np.ndarray
    images: np.ndarray
    wall_height: float
    horizontal_origin: np.ndarray
    span: float


def prepare(sources=None, targets=None, wall_height=0.0):
    """Translate and reflect geometry; with no arguments run a deterministic self-test."""
    if sources is None and targets is None:
        g = prepare([[0], [0], [2]], [[1, 2], [0, 0], [1, 3]], 1)
        np.testing.assert_array_equal(g.images[2], [-1])
        assert g.targets.shape == (3, 2) and not g.sources.flags.writeable
        return None
    wall_height = _scalar(wall_height, "wall_height")
    y, x = _array(sources, "sources"), _array(targets, "targets")
    if np.any(y[2] <= wall_height) or np.any(x[2] < wall_height):
        raise ValueError("Sources must be above and targets on or above the wall")
    # Compare original coordinates before translation can change their equality.
    source_keys = {tuple(p) for p in y.T}
    if any(tuple(p) in source_keys for p in x.T):
        raise SingularInteractionError("A physical source coincides with a target")
    with np.errstate(over="raise", invalid="raise"):
        y[2] -= wall_height
        x[2] -= wall_height
        points = np.concatenate((y, x), axis=1)
        origin = np.zeros(2)
        if points.shape[1]:
            origin = points[:2].min(axis=1) / 2 + points[:2].max(axis=1) / 2
            y[:2] -= origin[:, None]
            x[:2] -= origin[:, None]
        images = y.copy(order="F")
        images[2] *= -1
        points = np.concatenate((y, images, x), axis=1)
        span = float(np.ptp(points, axis=1).max()) if points.shape[1] else 0.0
    for array in (y, x, images, origin):
        array.setflags(write=False)
    return Geometry(y, x, images, wall_height, origin, span)


def guard(geom=None):
    """Reject backend-unresolved pairs using occupied spatial bins, without dense distances."""
    if geom is None:
        g = prepare([[0, 1], [0, 1], [1e-16, 1]], [[0], [0], [0]])
        try:
            guard(g)
        except FMMResolutionError:
            return None
        raise AssertionError("The FMM proximity guard failed")
    if not geom.sources.shape[1] or not geom.targets.shape[1]:
        return None
    tau = 64 * np.finfo(float).eps
    if not np.isfinite(geom.span) or geom.span <= 0:
        raise FMMResolutionError("The geometry span is not representable")
    # Normalize first to avoid underflow in a dimensional exclusion threshold.
    y = np.concatenate((geom.sources, geom.images), axis=1) / geom.span
    x = geom.targets / geom.span
    origin = np.minimum(y.min(axis=1), x.min(axis=1))
    ybins = np.floor((y - origin[:, None]) / (2 * tau)).astype(np.int64)
    xbins = np.floor((x - origin[:, None]) / (2 * tau)).astype(np.int64)
    bins = {}
    for n, cell in enumerate(ybins.T):
        bins.setdefault(tuple(cell), []).append(n)
    offsets = tuple(product((-1, 0, 1), repeat=3))
    ns = geom.sources.shape[1]
    for m, cell in enumerate(xbins.T):
        for offset in offsets:
            candidates = bins.get(tuple(cell + offset), ())
            for start in range(0, len(candidates), 512):
                subset = candidates[start:start + 512]
                distances = (y[:, subset] - x[:, m, None]) / tau
                hits = np.flatnonzero(np.sum(distances * distances, axis=0) <= 1)
                if hits.size:
                    n = subset[hits[0]]
                    kind = "physical" if n < ns else "reflected"
                    raise FMMResolutionError(
                        f"Target {m} is within 64*eps*span of {kind} source {n % ns}; "
                        "use direct evaluation"
                    )
    return None
