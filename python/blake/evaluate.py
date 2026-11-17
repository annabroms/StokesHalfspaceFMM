"""Public physical-velocity interface; FMM3D is imported only for nonempty FMM work."""

from time import perf_counter

import numpy as np

from .direct import _block_sizes, direct_velocity
from .fmm import fmm_velocity
from .geometry import Geometry, _as_scalar, _as_vector_array, prepare


def apply(
    geom=None,
    forces=None,
    *,
    method="fmm",
    mu=1.0,
    tol=1e-9,
    block_size=(256, 512),
    correct_near_wall=False,
    near_wall_distance=None,
    diagnostics=False,
    return_info=False,
):
    """Evaluate prepared geometry with FMM or independent direct Blake sums.

    correct_near_wall=True routes ill-conditioned close targets to direct summation.
    """
    if geom is None and forces is None:
        geometry = prepare([[0], [0], [1]], [[0.2], [0.4], [2]])
        forces = np.array([[1], [2], [3]])
        value = apply(geometry, forces, method="blake")
        np.testing.assert_allclose(value, direct_velocity(geometry, forces))
        shallow = prepare([[0, 0.5], [0, 0.5], [1e-6, 1]], [[0], [0], [0]])
        strengths = np.repeat(forces, 2, axis=1)
        corrected, info = apply(shallow, strengths, correct_near_wall=True, return_info=True)
        np.testing.assert_array_equal(corrected, direct_velocity(shallow, strengths))
        assert info["corrected_target_count"] == 1 and info["fmm_call_count"] == 0
        return None

    timer = perf_counter()
    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    forces = _as_vector_array(forces, "forces")
    if forces.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _as_scalar(mu, "mu", positive=True)
    tol = _as_scalar(tol, "tol", positive=True)
    if tol >= 1 or method not in ("fmm", "blake"):
        raise ValueError("Require 0 < tol < 1 and method='fmm' or 'blake'")
    block_size = _block_sizes(block_size)
    if not isinstance(correct_near_wall, (bool, np.bool_)):
        raise ValueError("correct_near_wall must be a boolean")
    if near_wall_distance is not None:
        near_wall_distance = _as_scalar(
            near_wall_distance, "near_wall_distance", positive=True
        )
    info = {
        "method": method,
        "ns": forces.shape[1],
        "nt": geom.targets.shape[1],
        "requested_tolerance": tol,
        "mu": mu,
        "wall_height": geom.wall_height,
        "geometry_seconds": 0.0,
        "direct_seconds": 0.0,
        "diagnostics_seconds": 0.0,
        "fmm_call_count": 0,
        "tree_reused": False,
        "correct_near_wall": bool(correct_near_wall),
        "corrected_target_count": 0,
        "near_wall_relative_radius": None,
        "near_wall_distance": None,
        "near_wall_detection_seconds": 0.0,
        "near_wall_direct_seconds": 0.0,
    }

    if method == "fmm":
        if correct_near_wall:
            from .near_wall import select_targets, target_subset

            start = perf_counter()
            selected, radius = select_targets(geom, tol=tol, distance=near_wall_distance)
            info["near_wall_relative_radius"] = radius
            info["near_wall_distance"] = radius * geom.span
            info["corrected_target_count"] = int(np.count_nonzero(selected))
            info["near_wall_detection_seconds"] = perf_counter() - start
        else:
            selected = None

        if selected is not None and np.any(selected):
            far_field, timing = fmm_velocity(
                target_subset(geom, ~selected), forces, mu=mu, tol=tol
            )
            start = perf_counter()
            near_field = direct_velocity(
                target_subset(geom, selected), forces, mu=mu, block_size=block_size
            )
            info["near_wall_direct_seconds"] = perf_counter() - start
            info["direct_seconds"] = info["near_wall_direct_seconds"]
            field = np.empty_like(geom.targets)
            field[:, ~selected] = far_field
            field[:, selected] = near_field
        else:
            field, timing = fmm_velocity(geom, forces, mu=mu, tol=tol)
        info.update(timing)
    else:
        start = perf_counter()
        field = direct_velocity(geom, forces, mu=mu, block_size=block_size)
        info["direct_seconds"] = perf_counter() - start
    if diagnostics:
        from .diagnostics import constituent_scale

        start = perf_counter()
        info["constituent_scale"] = constituent_scale(
            geom, forces, mu=mu, block_size=block_size
        )
        info["diagnostics_seconds"] = perf_counter() - start

    info["total_seconds"] = perf_counter() - timer
    if return_info:
        return field, info
    return field


def velocity(sources=None, forces=None, targets=None, *, wall_height=0.0, **options):
    """Evaluate physical Blake velocity with component-first (3, N) arrays."""
    if sources is None and forces is None and targets is None:
        sources = [[0], [0], [1]]
        forces = [[1], [2], [3]]
        targets = [[0.2], [0.4], [2]]
        value = velocity(sources, forces, targets, method="blake")
        reference = apply(prepare(sources, targets), forces, method="blake")
        np.testing.assert_allclose(value, reference)
        return None

    timer = perf_counter()
    geometry = prepare(sources, targets, wall_height)
    geometry_seconds = perf_counter() - timer
    result = apply(geometry, forces, **options)
    if options.get("return_info", False):
        field, info = result
        info["geometry_seconds"] = geometry_seconds
        info["total_seconds"] = perf_counter() - timer
        return field, info
    return result
