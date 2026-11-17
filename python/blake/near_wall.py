"""Geometric selection for optional direct replacement of ill-conditioned targets."""

from itertools import product

import numpy as np

from .geometry import FMMResolutionError, Geometry, _as_scalar, prepare


def select_targets(geom=None, *, tol=1e-9, distance=None):
    """Select close targets in linear storage; no arguments run a deterministic test."""
    if geom is None:
        geometry = prepare([[0], [0], [1e-6]], [[0, 0.4], [0, 0], [0, 1]])
        selected, radius = select_targets(geometry)
        np.testing.assert_array_equal(selected, [True, False])
        assert radius > 0
        return None

    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    tol = _as_scalar(tol, "tol", positive=True)
    if tol >= 1:
        raise ValueError("tol must be less than 1")
    if distance is not None:
        distance = _as_scalar(distance, "near_wall_distance", positive=True)
    machine_epsilon = np.finfo(float).eps
    # Roundoff in global-coordinate Stokes reconstruction can grow as eps*L/d.
    # This safety factor is a geometric heuristic, not a field error bound.
    # A radius of 2*span already covers the entire 3D bounding box.
    relative_radius = (
        2.0 if tol <= 4 * machine_epsilon
        else max(64 * machine_epsilon, 8 * machine_epsilon / tol)
    )
    selected = np.zeros(geom.targets.shape[1], dtype=bool)
    if not geom.sources.shape[1] or not selected.size:
        return selected, relative_radius
    if not np.isfinite(geom.span) or geom.span <= 0:
        raise FMMResolutionError("Geometry span must be finite and positive")
    if distance is not None:
        relative_radius = max(64 * machine_epsilon, min(2.0, distance / geom.span))

    sources = geom.sources / geom.span
    targets = geom.targets / geom.span
    source_cells = np.floor(sources / relative_radius).astype(np.int64)
    target_cells = np.floor(targets / relative_radius).astype(np.int64)
    occupied_cells = {}
    for source_index, cell in enumerate(source_cells.T):
        occupied_cells.setdefault(tuple(cell), []).append(source_index)

    # Image distances cannot be smaller than physical distances above the wall.
    # Check neighboring occupied cells, stopping after the first close source.
    neighbor_offsets = tuple(product((-1, 0, 1), repeat=3))
    for target_index, (cx, cy, cz) in enumerate(target_cells.T):
        for dx, dy, dz in neighbor_offsets:
            candidates = occupied_cells.get((cx + dx, cy + dy, cz + dz), ())
            for start in range(0, len(candidates), 512):
                subset = candidates[start : start + 512]
                separation = (sources[:, subset] - targets[:, target_index, None])
                separation /= relative_radius
                if np.any(np.sum(separation * separation, axis=0) <= 1):
                    selected[target_index] = True
                    break
            if selected[target_index]:
                break
    return selected, relative_radius


def target_subset(geom, selected):
    """Restrict targets without translating coordinates or changing the selection scale."""
    targets = np.asfortranarray(geom.targets[:, selected])
    targets.setflags(write=False)
    return Geometry(
        geom.sources, targets, geom.images, geom.wall_height,
        geom.horizontal_origin, geom.span,
    )
