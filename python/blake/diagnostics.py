"""Opt-in blocked constituent scales and cancellation-aware velocity metrics."""

import numpy as np

from .direct import _block_sizes
from .geometry import Geometry, _as_scalar, _as_vector_array, prepare


def constituent_scale(geom=None, forces=None, *, mu=1.0, block_size=(256, 512)):
    """Sum magnitudes before cancellation; O(Ns*Nt) time with bounded workspace."""
    if geom is None and forces is None:
        geometry = prepare([[0], [0], [1]], [[1], [0], [0]])
        scale = constituent_scale(geometry, [[1], [2], [3]])
        assert scale.shape == (1,) and scale[0] > 0
        zero_scale = constituent_scale(geometry, np.zeros((3, 1)))
        np.testing.assert_array_equal(zero_scale, [0])
        return None

    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    forces = _as_vector_array(forces, "forces")
    if forces.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _as_scalar(mu, "mu", positive=True)
    target_block_size, source_block_size = _block_sizes(block_size)
    total_scale = np.zeros(geom.targets.shape[1])

    for target_start in range(0, total_scale.size, target_block_size):
        target_slice = slice(target_start, target_start + target_block_size)
        for source_start in range(0, forces.shape[1], source_block_size):
            source_slice = slice(source_start, source_start + source_block_size)

            # Pair workspaces have axes (component, target, source).
            block_forces = forces[:, None, source_slice]
            reflected_forces = block_forces.copy()
            reflected_forces[2] *= -1
            r = geom.targets[:, target_slice, None] - geom.sources[:, None, source_slice]
            r_image = (
                geom.targets[:, target_slice, None] - geom.images[:, None, source_slice]
            )
            inverse_r = 1 / np.linalg.norm(r, axis=0)
            inverse_image_r = 1 / np.linalg.norm(r_image, axis=0)

            # Sum constituent magnitudes before the wall correction cancels them.
            stokes_velocity = (
                block_forces * inverse_r
                + r * np.sum(r * block_forces, axis=0) * inverse_r**3
            ) / (8 * np.pi)
            image_velocity = (
                reflected_forces * inverse_image_r
                + r_image * np.sum(r_image * reflected_forces, axis=0) * inverse_image_r**3
            ) / (8 * np.pi)

            charges = reflected_forces[2]
            dipoles = reflected_forces * geom.sources[2, source_slice]
            image_r_dot_dipole = np.sum(r_image * dipoles, axis=0)
            potential = (
                charges * inverse_image_r + image_r_dot_dipole * inverse_image_r**3
            ) / (4 * np.pi)
            gradient = (
                -charges * r_image * inverse_image_r**3
                + dipoles * inverse_image_r**3
                - 3 * r_image * image_r_dot_dipole * inverse_image_r**5
            ) / (4 * np.pi)

            target_heights = geom.targets[2, target_slice, None]
            pair_scale = np.linalg.norm(stokes_velocity, axis=0)
            pair_scale += np.linalg.norm(image_velocity, axis=0)
            pair_scale += (
                target_heights * np.linalg.norm(gradient, axis=0) + np.abs(potential)
            )
            total_scale[target_slice] += np.sum(pair_scale, axis=1) / mu

    if not np.isfinite(total_scale).all():
        raise FloatingPointError("Constituent scale overflowed; rescale inputs")
    return total_scale


def metrics(value=None, reference=None, scale=None):
    """Report absolute, global relative and constituent-normalized velocity errors."""
    if value is None and reference is None and scale is None:
        errors = metrics(np.zeros((3, 1)), np.zeros((3, 1)), np.ones(1))
        assert np.isnan(errors["global_relative"])
        assert errors["max_absolute"] == 0
        return None

    value = _as_vector_array(value, "value")
    reference = _as_vector_array(reference, "reference")
    scale = np.asarray(scale, dtype=float)
    if value.shape != reference.shape or scale.shape != (value.shape[1],):
        raise ValueError("Expected matching (3, Nt) fields and (Nt,) scale")
    if not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError("scale must be finite and nonnegative")

    pointwise_error = np.linalg.norm(value - reference, axis=0)
    normalized_error = np.zeros_like(pointwise_error)
    np.divide(pointwise_error, scale, out=normalized_error, where=scale > 0)
    normalized_error[(scale == 0) & (pointwise_error > 0)] = np.inf

    # A zero or negligible reference, such as wall velocity, has no relative error.
    reference_norm = np.linalg.norm(reference)
    roundoff_scale = 100 * np.finfo(float).eps * np.linalg.norm(scale)
    relative_error = float("nan")
    if reference_norm > roundoff_scale:
        relative_error = float(np.linalg.norm(value - reference) / reference_norm)

    return {
        "max_absolute": float(pointwise_error.max(initial=0)),
        "global_relative": relative_error,
        "constituent_normalized": float(normalized_error.max(initial=0)),
    }
