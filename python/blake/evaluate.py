"""Public physical-field interface; FMM3D is imported only for nonempty FMM work."""

from time import perf_counter

import numpy as np

from .direct import _block_sizes, direct_fields, direct_velocity
from .fmm import fmm_fields
from .geometry import Geometry, _as_scalar, _as_vector_array, prepare

FIELD_NAMES = ("velocity", "pressure", "gradient", "traction")


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

    fields, info = _evaluate(
        geom, forces, 1, method=method, mu=mu, tol=tol, block_size=block_size,
        correct_near_wall=correct_near_wall, near_wall_distance=near_wall_distance,
        diagnostics=diagnostics,
    )
    if return_info:
        return fields["velocity"], info
    return fields["velocity"]


def apply_fields(
    geom=None,
    forces=None,
    *,
    requested=("velocity",),
    normals=None,
    method="fmm",
    mu=1.0,
    tol=1e-9,
    block_size=(256, 512),
    correct_near_wall=False,
    near_wall_distance=None,
    diagnostics=False,
    return_info=False,
):
    """Evaluate velocity plus requested pressure, gradient or traction on prepared geometry.

    Returns a dict: velocity (3, Nt); pressure (Nt,), not divided by mu;
    grad_u (3, 3, Nt) with grad_u[i, j, m] = du_i/dx_j; traction (3, Nt) for
    unit normals (3, Nt), traction = (-p I + mu (grad_u + grad_u^T)) n.
    """
    if geom is None and forces is None:
        geometry = prepare([[0, 0.4], [0, 0.1], [1, 0.3]], [[0.2, 0.3], [0.4, 0.1], [0, 2]])
        forces = np.array([[1, -1], [2, 0.5], [3, 2]])
        normals = np.array([[0, 0.6], [0, 0], [1, 0.8]])
        value = apply_fields(geometry, forces, requested="traction", normals=normals,
                             method="blake", mu=1.7)
        reference = direct_fields(geometry, forces, mu=1.7)
        for m in range(2):
            G, n = reference["grad_u"][:, :, m], normals[:, m]
            expected = -reference["pressure"][m] * n + 1.7 * (G + G.T) @ n
            np.testing.assert_allclose(value["traction"][:, m], expected, atol=1e-14)
        assert set(value) == {"velocity", "pressure", "grad_u", "traction"}
        return None

    requested = _requested_fields(requested)
    order = 1
    if "pressure" in requested:
        order = 2
    if "gradient" in requested or "traction" in requested:
        order = 3
    nt = geom.targets.shape[1] if isinstance(geom, Geometry) else None
    if "traction" in requested:
        normals = _unit_normals(normals, nt)
    elif normals is not None:
        raise ValueError("normals are used only when traction is requested")

    timer = perf_counter()
    fields, info = _evaluate(
        geom, forces, order, method=method, mu=mu, tol=tol, block_size=block_size,
        correct_near_wall=correct_near_wall, near_wall_distance=near_wall_distance,
        diagnostics=diagnostics,
    )
    info["requested"] = list(requested)
    if "traction" in requested:
        start = perf_counter()
        grad_u = fields["grad_u"]
        strain_normal = (np.einsum("ijm,jm->im", grad_u, normals)
                         + np.einsum("jim,jm->im", grad_u, normals))
        fields["traction"] = info["mu"] * strain_normal - fields["pressure"] * normals
        if not np.isfinite(fields["traction"]).all():
            raise FloatingPointError("Traction overflowed; rescale inputs")
        info["assembly_seconds"] = info.get("assembly_seconds", 0.0) + perf_counter() - start
        info["total_seconds"] = perf_counter() - timer
    if return_info:
        return fields, info
    return fields


def _requested_fields(requested):
    """Validate requested field names; a single name may be given as a string."""
    if isinstance(requested, str):
        requested = (requested,)
    try:
        names = tuple(requested)
    except TypeError as exc:
        raise ValueError(f"requested must contain names from {FIELD_NAMES}") from exc
    if not names or any(not isinstance(n, str) or n not in FIELD_NAMES for n in names):
        raise ValueError(f"requested must contain names from {FIELD_NAMES}")
    return tuple(dict.fromkeys(names))


def _unit_normals(normals, target_count):
    """Require finite unit normals of shape (3, Nt) for traction."""
    if normals is None:
        raise ValueError("Traction requires unit normals with shape (3, Nt)")
    normals = _as_vector_array(normals, "normals")
    if target_count is not None and normals.shape[1] != target_count:
        raise ValueError("normals must have shape (3, Nt)")
    if np.any(np.abs(np.linalg.norm(normals, axis=0) - 1) > 1e-12):
        raise ValueError("Traction normals must have unit length")
    return normals


def _evaluate(geom, forces, order, *, method, mu, tol, block_size,
              correct_near_wall, near_wall_distance, diagnostics):
    """Shared validation and dispatch for velocity (order 1), pressure (2) and gradients (3)."""
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
        "field_order": order,
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
            far_fields, timing = fmm_fields(
                target_subset(geom, ~selected), forces, order=order, mu=mu, tol=tol
            )
            start = perf_counter()
            near_fields = direct_fields(
                target_subset(geom, selected), forces, order=order, mu=mu,
                block_size=block_size,
            )
            info["near_wall_direct_seconds"] = perf_counter() - start
            info["direct_seconds"] = info["near_wall_direct_seconds"]
            # Targets are the last axis of every field.
            fields = {}
            for name, far in far_fields.items():
                value = np.empty(far.shape[:-1] + (geom.targets.shape[1],))
                value[..., ~selected] = far
                value[..., selected] = near_fields[name]
                fields[name] = value
        else:
            fields, timing = fmm_fields(geom, forces, order=order, mu=mu, tol=tol)
        info.update(timing)
    else:
        start = perf_counter()
        fields = direct_fields(geom, forces, order=order, mu=mu, block_size=block_size)
        info["direct_seconds"] = perf_counter() - start
    if diagnostics:
        from .diagnostics import constituent_scale

        start = perf_counter()
        info["constituent_scale"] = constituent_scale(
            geom, forces, mu=mu, block_size=block_size
        )
        info["diagnostics_seconds"] = perf_counter() - start

    info["total_seconds"] = perf_counter() - timer
    return fields, info


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


def fields(sources=None, forces=None, targets=None, *, wall_height=0.0, **options):
    """Evaluate velocity plus requested pressure, gradient or traction from raw (3, N) arrays.

    Options are those of apply_fields, e.g. requested=("pressure", "traction"), normals.
    """
    if sources is None and forces is None and targets is None:
        sources, forces = [[0], [0], [1]], [[1], [2], [3]]
        targets = [[0.2, 0.1], [0.4, 0.3], [2, 0]]
        value = fields(sources, forces, targets, requested="gradient", method="blake")
        reference = apply_fields(prepare(sources, targets), forces,
                                 requested="gradient", method="blake")
        np.testing.assert_array_equal(value["grad_u"], reference["grad_u"])
        return None

    timer = perf_counter()
    geometry = prepare(sources, targets, wall_height)
    geometry_seconds = perf_counter() - timer
    result = apply_fields(geometry, forces, **options)
    if options.get("return_info", False):
        value, info = result
        info["geometry_seconds"] = geometry_seconds
        info["total_seconds"] = perf_counter() - timer
        return value, info
    return result
