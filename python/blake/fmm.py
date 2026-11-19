"""Two target-only FMM3D calls using Gimbutas–Greengard–Veerapaneni (2015).

Velocity, pressure and velocity gradients share the same two calls; only the
requested target flags change.
"""

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


# FMM3D packs symmetric Laplace Hessians as xx yy zz xy xz yz.
_HESSIAN_INDEX = np.array([[0, 3, 4], [3, 1, 5], [4, 5, 2]])


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

    fields, timing = fmm_fields(geom, forces, order=1, mu=mu, tol=tol)
    return fields["velocity"], timing


def fmm_fields(geom=None, forces=None, *, order=1, mu=1.0, tol=1e-9):
    """Evaluate velocity, pressure (order>=2) and grad_u (order 3) with two FMM calls.

    grad_u[i, j, m] = du_i/dx_j at target m; pressure is not divided by mu.
    """
    if geom is None and forces is None:
        from .direct import direct_fields

        geometry = prepare([[0, 0.4], [0, 0.1], [1, 0.3]], [[0.2, 0.3], [0.4, 0.1], [0, 2]])
        forces = np.array([[1, -1], [2, 0.5], [3, 2]])
        value, info = fmm_fields(geometry, forces, order=3, mu=1.7, tol=1e-12)
        reference = direct_fields(geometry, forces, mu=1.7)
        for name in ("velocity", "pressure", "grad_u"):
            np.testing.assert_allclose(value[name], reference[name], atol=1e-11)
        assert info["fmm_call_count"] == 2 and info["laplace_target_flag"] == 3
        return None

    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    if order not in (1, 2, 3):
        raise ValueError("order must be 1 (velocity), 2 (pressure) or 3 (gradient)")
    forces = _as_vector_array(forces, "forces")
    if forces.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _as_scalar(mu, "mu", positive=True)
    tol = _as_scalar(tol, "tol", positive=True)
    if tol >= 1:
        raise ValueError("tol must be less than 1")

    timing = {
        "guard_seconds": 0.0,
        "guard_backend": "numba-bins",
        "strength_seconds": 0.0,
        "stokes_seconds": 0.0,
        "laplace_seconds": 0.0,
        "assembly_seconds": 0.0,
        "fmm_call_count": 0,
        "tree_reused": False,
        "stokes_target_flag": 0,
        "laplace_target_flag": 0,
    }
    target_count = geom.targets.shape[1]
    if not geom.sources.shape[1] or not target_count:
        return _zero_fields(order, target_count), timing

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

    # Call 1: physical and image Stokeslets. Target flag 1 gives velocity,
    # 2 adds pressure and 3 adds the velocity gradient.
    timer = perf_counter()
    stokes = fmm3dpy.stfmm3d(
        eps=tol,
        sources=stokes_sources,
        stoklet=stokes_strengths,
        targets=geom.targets,
        ifppreg=0,
        ifppregtarg=order,
        nd=1,
    )
    timing["stokes_seconds"] = perf_counter() - timer
    timing["stokes_target_flag"] = order
    stokes_velocity = _checked_output(stokes, "pottarg", (3, target_count), "stfmm3d")

    # Call 2: image charges and dipoles. Velocity and pressure need phi and its
    # gradient (flag 2); velocity gradients also need its Hessian (flag 3).
    laplace_flag = max(2, order)
    timer = perf_counter()
    harmonic = fmm3dpy.lfmm3d(
        eps=tol,
        sources=geom.images,
        charges=harmonic_charges,
        dipvec=harmonic_dipoles,
        targets=geom.targets,
        pg=0,
        pgt=laplace_flag,
        nd=1,
    )
    timing["laplace_seconds"] = perf_counter() - timer
    timing["laplace_target_flag"] = laplace_flag

    timer = perf_counter()
    potential = _checked_output(harmonic, "pottarg", (target_count,), "lfmm3d")
    gradient = _checked_output(harmonic, "gradtarg", (3, target_count), "lfmm3d")

    # Complete physical field: u = (v - z*grad(phi) + e3*phi) / mu.
    # At the wall these same operations compute the residual without overwriting it.
    target_heights = geom.targets[2]
    velocity = stokes_velocity - target_heights * gradient
    velocity[2] += potential
    velocity /= mu
    fields = {"velocity": velocity}
    if order >= 2:
        # GGV (3), (17): the subtracted Papkovich-Neuber field carries pressure 2*dphi/dz.
        stokes_pressure = _checked_output(stokes, "pretarg", (target_count,), "stfmm3d")
        fields["pressure"] = stokes_pressure - 2 * gradient[2]
    if order >= 3:
        # fmm3dpy returns gradtarg[k, i, m] = dv_i/dx_k; transpose to [i, k, m].
        stokes_gradient = _checked_output(
            stokes, "gradtarg", (3, 3, target_count), "stfmm3d"
        ).transpose(1, 0, 2)
        packed = _checked_output(harmonic, "hesstarg", (6, target_count), "lfmm3d")
        hessian = packed[_HESSIAN_INDEX]
        # du_i/dx_j = [dv_i/dx_j - delta_j3*g_i - z*H_ij + delta_i3*g_j] / mu.
        grad_u = stokes_gradient - target_heights * hessian
        grad_u[:, 2, :] -= gradient
        grad_u[2, :, :] += gradient
        fields["grad_u"] = grad_u / mu
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise FloatingPointError("FMM fields overflowed; rescale inputs")
    timing["assembly_seconds"] = perf_counter() - timer
    timing["fmm_call_count"] = 2
    return fields, timing


def _zero_fields(order, target_count):
    """Return correctly shaped zero fields for empty source or target sets."""
    fields = {"velocity": np.zeros((3, target_count))}
    if order >= 2:
        fields["pressure"] = np.zeros(target_count)
    if order >= 3:
        fields["grad_u"] = np.zeros((3, 3, target_count))
    return fields
