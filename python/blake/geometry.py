"""Validated, translated geometry and bounded-memory singularity checks."""

from dataclasses import dataclass
from itertools import product

import numpy as np


class SingularInteractionError(ValueError):
    """An exact physical source/target coincidence has no finite velocity."""


class FMMResolutionError(ValueError):
    """The FMM exclusion radius cannot resolve a physical or image interaction."""


def _as_vector_array(value, name):
    """Validate a finite real component-first array and own its float64 storage."""
    raw = np.asarray(value)
    if raw.ndim != 2 or raw.shape[0] != 3 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a real numeric (3, N) array")
    array = np.array(raw, dtype=np.float64, order="F", copy=True)
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _as_scalar(value, name, positive=False):
    """Validate a finite real scalar without silently discarding imaginary parts."""
    raw = np.asarray(value)
    if raw.ndim != 0 or raw.dtype.kind not in "fiu":
        raise ValueError(f"{name} must be a finite real scalar")
    result = float(raw)
    if not np.isfinite(result) or (positive and result <= 0):
        requirement = "finite and positive" if positive else "finite"
        raise ValueError(f"{name} must be {requirement}")
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
        geometry = prepare([[0], [0], [2]], [[1, 2], [0, 0], [1, 3]], 1)
        np.testing.assert_array_equal(geometry.images[2], [-1])
        assert geometry.targets.shape == (3, 2)
        assert not geometry.sources.flags.writeable
        return None

    wall_height = _as_scalar(wall_height, "wall_height")
    sources = _as_vector_array(sources, "sources")
    targets = _as_vector_array(targets, "targets")
    if np.any(sources[2] <= wall_height) or np.any(targets[2] < wall_height):
        raise ValueError("Sources must be above and targets on or above the wall")

    # Compare original coordinates before translation can change their equality.
    source_keys = {tuple(point) for point in sources.T}
    if any(tuple(point) in source_keys for point in targets.T):
        raise SingularInteractionError("A physical source coincides with a target")

    with np.errstate(over="raise", invalid="raise"):
        # Put the wall at z=0, then center the horizontal coordinates together.
        sources[2] -= wall_height
        targets[2] -= wall_height
        points = np.concatenate((sources, targets), axis=1)
        horizontal_origin = np.zeros(2)
        if points.shape[1]:
            horizontal_origin = (
                points[:2].min(axis=1) / 2 + points[:2].max(axis=1) / 2
            )
            sources[:2] -= horizontal_origin[:, None]
            targets[:2] -= horizontal_origin[:, None]

        images = sources.copy(order="F")
        images[2] *= -1
        all_points = np.concatenate((sources, images, targets), axis=1)
        span = 0.0
        if all_points.shape[1]:
            span = float(np.ptp(all_points, axis=1).max())

    for array in (sources, targets, images, horizontal_origin):
        array.setflags(write=False)
    return Geometry(sources, targets, images, wall_height, horizontal_origin, span)


def guard(geom=None):
    """Reject unresolved FMM pairs using occupied bins and bounded distance checks."""
    if geom is None:
        geometry = prepare([[0, 1], [0, 1], [1e-16, 1]], [[0], [0], [0]])
        try:
            guard(geometry)
        except FMMResolutionError:
            return None
        raise AssertionError("The FMM proximity guard failed")
    if not geom.sources.shape[1] or not geom.targets.shape[1]:
        return None
    relative_threshold = 64 * np.finfo(float).eps
    if not np.isfinite(geom.span) or geom.span <= 0:
        raise FMMResolutionError("The geometry span is not representable")
    # Normalize first to avoid underflow in a dimensional exclusion threshold.
    sources = np.concatenate((geom.sources, geom.images), axis=1) / geom.span
    targets = geom.targets / geom.span
    origin = np.minimum(sources.min(axis=1), targets.min(axis=1))
    cell_width = 2 * relative_threshold
    source_cells = np.floor((sources - origin[:, None]) / cell_width).astype(np.int64)
    target_cells = np.floor((targets - origin[:, None]) / cell_width).astype(np.int64)

    # Store only occupied cells; each target searches its 27 neighboring cells.
    occupied_cells = {}
    for source_index, cell in enumerate(source_cells.T):
        occupied_cells.setdefault(tuple(cell), []).append(source_index)

    neighbor_offsets = tuple(product((-1, 0, 1), repeat=3))
    source_count = geom.sources.shape[1]
    for target_index, cell in enumerate(target_cells.T):
        for offset in neighbor_offsets:
            candidates = occupied_cells.get(tuple(cell + offset), ())
            for start in range(0, len(candidates), 512):
                subset = candidates[start : start + 512]
                separation = sources[:, subset] - targets[:, target_index, None]
                scaled_separation = separation / relative_threshold
                distance_squared = np.sum(scaled_separation * scaled_separation, axis=0)
                hits = np.flatnonzero(distance_squared <= 1)
                if hits.size:
                    source_index = subset[hits[0]]
                    kind = "physical" if source_index < source_count else "reflected"
                    physical_index = source_index % source_count
                    raise FMMResolutionError(
                        f"Target {target_index} is within 64*eps*span "
                        f"of {kind} source {physical_index}; "
                        "use direct evaluation"
                    )
    return None
