"""Two target-only FMM3D calls using Gimbutas–Greengard–Veerapaneni (2015)."""

from time import perf_counter

import numpy as np

from .geometry import Geometry, _as_scalar, _as_vector_array, guard, prepare


def _checked_output(out, name, shape, backend):
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
    """Evaluate physical velocity with one Stokes call and one Laplace call."""
    if geom is None and forces is None:
        from .direct import direct_velocity

        geometry = prepare([[0], [0], [1]], [[0.2, 0.3], [0.4, 0.1], [0, 2]])
        forces = np.array([[1], [2], [3]])
        value, info = fmm_velocity(geometry, forces, tol=1e-12)
        reference = direct_velocity(geometry, forces)
        np.testing.assert_allclose(value, reference, atol=1e-12)
        assert info["fmm_call_count"] == 2
        return None

    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    forces = _as_vector_array(forces, "forces")
    if forces.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _as_scalar(mu, "mu", positive=True)
    tol = _as_scalar(tol, "tol", positive=True)
    if tol >= 1:
        raise ValueError("tol must be less than 1")

    timing = {
        "guard_seconds": 0.0,
        "strength_seconds": 0.0,
        "stokes_seconds": 0.0,
        "laplace_seconds": 0.0,
        "assembly_seconds": 0.0,
        "fmm_call_count": 0,
        "tree_reused": False,
    }
    target_count = geom.targets.shape[1]
    if not geom.sources.shape[1] or not target_count:
        return np.zeros((3, target_count)), timing

    timer = perf_counter()
    guard(geom)
    timing["guard_seconds"] = perf_counter() - timer

    try:
        import fmm3dpy
    except ImportError as exc:
        raise ImportError(
            "FMM3D unavailable: select the interpreter containing fmm3dpy"
        ) from exc

    # Reflect f to f*=(f1,f2,-f3). The image Stokeslet has strength -f*.
    timer = perf_counter()
    reflected_forces = forces.copy(order="F")
    reflected_forces[2] *= -1
    stokes_sources = np.asfortranarray(
        np.concatenate((geom.sources, geom.images), axis=1)
    )
    stokes_strengths = np.asfortranarray(
        np.concatenate((forces, -reflected_forces), axis=1)
    )
    harmonic_charges = reflected_forces[2].copy()
    harmonic_dipoles = np.asfortranarray(reflected_forces * geom.sources[2])
    timing["strength_seconds"] = perf_counter() - timer

    # Call 1: physical and image Stokeslets, requesting target velocity only.
    timer = perf_counter()
    stokes = fmm3dpy.stfmm3d(
        eps=tol,
        sources=stokes_sources,
        stoklet=stokes_strengths,
        targets=geom.targets,
        ifppreg=0,
        ifppregtarg=1,
        nd=1,
    )
    timing["stokes_seconds"] = perf_counter() - timer
    stokes_velocity = _checked_output(stokes, "pottarg", (3, target_count), "stfmm3d")

    # Call 2: image charges and dipoles, requesting phi and its target gradient.
    timer = perf_counter()
    harmonic = fmm3dpy.lfmm3d(
        eps=tol,
        sources=geom.images,
        charges=harmonic_charges,
        dipvec=harmonic_dipoles,
        targets=geom.targets,
        pg=0,
        pgt=2,
        nd=1,
    )
    timing["laplace_seconds"] = perf_counter() - timer

    timer = perf_counter()
    potential = _checked_output(harmonic, "pottarg", (target_count,), "lfmm3d")
    gradient = _checked_output(harmonic, "gradtarg", (3, target_count), "lfmm3d")

    # Complete physical field: u = (v - z*grad(phi) + e3*phi) / mu.
    # At the wall these same operations compute the residual without overwriting it.
    target_heights = geom.targets[2]
    velocity = stokes_velocity - target_heights * gradient
    velocity[2] += potential
    velocity /= mu
    if not np.isfinite(velocity).all():
        raise FloatingPointError("FMM velocity overflowed; rescale inputs")
    timing["assembly_seconds"] = perf_counter() - timer
    timing["fmm_call_count"] = 2
    return velocity, timing
