"""Public physical-velocity interface; FMM3D is imported only for nonempty FMM work."""

from time import perf_counter

import numpy as np

from .direct import _blocks, direct_velocity
from .fmm import fmm_velocity
from .geometry import Geometry, _array, _scalar, prepare


def apply(geom=None, forces=None, *, method="fmm", mu=1.0, tol=1e-9,
          block_size=(256, 512), diagnostics=False, return_info=False):
    """Evaluate prepared geometry; select method='blake' for the independent direct sum."""
    if geom is None and forces is None:
        g = prepare([[0], [0], [1]], [[.2], [.4], [2]])
        f = np.array([[1], [2], [3]])
        np.testing.assert_allclose(apply(g, f, method="blake"), direct_velocity(g, f))
        return None
    timer = perf_counter()
    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    f = _array(forces, "forces")
    if f.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu, tol = _scalar(mu, "mu", True), _scalar(tol, "tol", True)
    if tol >= 1 or method not in ("fmm", "blake"):
        raise ValueError("Require 0 < tol < 1 and method='fmm' or 'blake'")
    block_size = _blocks(block_size)
    info = dict(method=method, ns=f.shape[1], nt=geom.targets.shape[1],
                requested_tolerance=tol, mu=mu, wall_height=geom.wall_height,
                geometry_seconds=0.0, direct_seconds=0.0, diagnostics_seconds=0.0,
                fmm_call_count=0, tree_reused=False)
    if method == "fmm":
        u, timing = fmm_velocity(geom, f, mu=mu, tol=tol)
        info.update(timing)
    else:
        start = perf_counter()
        u = direct_velocity(geom, f, mu=mu, block_size=block_size)
        info["direct_seconds"] = perf_counter() - start
    if diagnostics:
        from .diagnostics import constituent_scale
        start = perf_counter()
        info["constituent_scale"] = constituent_scale(geom, f, mu=mu, block_size=block_size)
        info["diagnostics_seconds"] = perf_counter() - start
    info["total_seconds"] = perf_counter() - timer
    return (u, info) if return_info else u


def velocity(sources=None, forces=None, targets=None, *, wall_height=0.0, **options):
    """Evaluate physical Blake velocity with component-first (3, N) arrays."""
    if sources is None and forces is None and targets is None:
        y, f, x = [[0], [0], [1]], [[1], [2], [3]], [[.2], [.4], [2]]
        np.testing.assert_allclose(velocity(y, f, x, method="blake"),
                                   apply(prepare(y, x), f, method="blake"))
        return None
    timer = perf_counter()
    geom = prepare(sources, targets, wall_height)
    geometry_seconds = perf_counter() - timer
    result = apply(geom, forces, **options)
    if options.get("return_info", False):
        u, info = result
        info["geometry_seconds"] = geometry_seconds
        info["total_seconds"] = perf_counter() - timer
        return u, info
    return result
