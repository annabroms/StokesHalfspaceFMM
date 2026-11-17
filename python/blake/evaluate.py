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
    diagnostics=False,
    return_info=False,
):
    """Evaluate prepared geometry; select method='blake' for the independent direct sum."""
    if geom is None and forces is None:
        geometry = prepare([[0], [0], [1]], [[0.2], [0.4], [2]])
        forces = np.array([[1], [2], [3]])
        value = apply(geometry, forces, method="blake")
        np.testing.assert_allclose(value, direct_velocity(geometry, forces))
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
    }

    if method == "fmm":
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
