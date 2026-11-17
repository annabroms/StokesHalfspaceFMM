"""Two target-only FMM3D calls using Gimbutas–Greengard–Veerapaneni (2015)."""

from time import perf_counter

import numpy as np

from .geometry import Geometry, _array, _scalar, guard, prepare


def _output(out, name, shape, backend):
    """Reject a failed status, missing field, malformed shape or nonfinite FMM output."""
    if getattr(out, "ier", None) != 0:
        raise RuntimeError(f"{backend} returned ier={getattr(out, 'ier', None)}")
    value = getattr(out, name, None)
    if value is None:
        raise RuntimeError(f"{backend} did not return {name}")
    array = np.asarray(value)
    # nd=1 wrappers may retain or squeeze the singleton density dimension.
    if array.shape not in (shape, (1,) + shape):
        raise RuntimeError(f"{backend} returned {name} with unexpected shape {array.shape}")
    if not np.isrealobj(array) or not np.isfinite(array).all():
        raise RuntimeError(f"{backend} returned invalid {name}")
    return array.reshape(shape)


def fmm_velocity(geom=None, forces=None, *, mu=1.0, tol=1e-9):
    """Return physical velocity and timing metadata using one Stokes and one Laplace call."""
    if geom is None and forces is None:
        from .direct import direct_velocity
        g = prepare([[0], [0], [1]], [[.2, .3], [.4, .1], [0, 2]])
        f = np.array([[1], [2], [3]])
        u, info = fmm_velocity(g, f, tol=1e-12)
        np.testing.assert_allclose(u, direct_velocity(g, f), atol=1e-12)
        assert info["fmm_call_count"] == 2
        return None
    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    f = _array(forces, "forces")
    if f.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu, tol = _scalar(mu, "mu", True), _scalar(tol, "tol", True)
    if tol >= 1:
        raise ValueError("tol must be less than 1")
    timing = dict(guard_seconds=0.0, strength_seconds=0.0, stokes_seconds=0.0,
                  laplace_seconds=0.0, assembly_seconds=0.0, fmm_call_count=0,
                  tree_reused=False)
    nt = geom.targets.shape[1]
    if not geom.sources.shape[1] or not nt:
        return np.zeros((3, nt)), timing
    timer = perf_counter()
    guard(geom)
    timing["guard_seconds"] = perf_counter() - timer
    try:
        import fmm3dpy
    except ImportError as exc:
        raise ImportError("FMM3D unavailable: select the interpreter containing fmm3dpy") from exc
    timer = perf_counter()
    reflected = f.copy(order="F")
    reflected[2] *= -1
    sources = np.asfortranarray(np.concatenate((geom.sources, geom.images), axis=1))
    stoklet = np.asfortranarray(np.concatenate((f, -reflected), axis=1))
    charges = reflected[2].copy()
    dipvec = np.asfortranarray(reflected * geom.sources[2])
    timing["strength_seconds"] = perf_counter() - timer
    timer = perf_counter()
    stokes = fmm3dpy.stfmm3d(eps=tol, sources=sources, stoklet=stoklet,
                           targets=geom.targets, ifppreg=0, ifppregtarg=1, nd=1)
    timing["stokes_seconds"] = perf_counter() - timer
    v = _output(stokes, "pottarg", (3, nt), "stfmm3d")
    timer = perf_counter()
    harmonic = fmm3dpy.lfmm3d(eps=tol, sources=geom.images, charges=charges,
                            dipvec=dipvec, targets=geom.targets, pg=0, pgt=2, nd=1)
    timing["laplace_seconds"] = perf_counter() - timer
    timer = perf_counter()
    phi = _output(harmonic, "pottarg", (nt,), "lfmm3d")
    gradient = _output(harmonic, "gradtarg", (3, nt), "lfmm3d")
    # Subtract the complete PN velocity; do not overwrite wall residuals.
    u = v - geom.targets[2] * gradient
    u[2] += phi
    u /= mu
    if not np.isfinite(u).all():
        raise FloatingPointError("FMM velocity overflowed; rescale inputs")
    timing["assembly_seconds"] = perf_counter() - timer
    timing["fmm_call_count"] = 2
    return u, timing
