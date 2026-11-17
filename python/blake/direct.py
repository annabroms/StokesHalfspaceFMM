"""Independent Blake (1971), equation (16); no Papkovich–Neuber assembly."""

import numpy as np

from .geometry import (
    Geometry,
    SingularInteractionError,
    _as_scalar,
    _as_vector_array,
    prepare,
)


def _block_sizes(block_size):
    """Validate target/source block sizes before allocating interaction workspaces."""
    values = np.asarray(block_size)
    if values.shape != (2,) or values.dtype.kind not in "iu" or np.any(values <= 0):
        raise ValueError("block_size must contain two positive integers (targets, sources)")
    return tuple(int(v) for v in values)


def blake_pair(source=None, target=None, *, mu=1.0, wall_height=0.0):
    """Return the (3, 3) Blake velocity tensor for (3, 1) source and target arrays."""
    if source is None and target is None:
        source = np.array([[0.1], [-0.2], [0.7]])
        target = np.array([[0.4], [0.6], [1.2]])
        forward = blake_pair(source, target)
        reverse = blake_pair(target, source)
        np.testing.assert_allclose(forward, reverse.T, atol=1e-14)
        wall_tensor = blake_pair(source, [[0.4], [0.6], [0]])
        np.testing.assert_allclose(wall_tensor, 0, atol=1e-14)
        return None

    geometry = prepare(source, target, wall_height)
    if geometry.sources.shape != (3, 1) or geometry.targets.shape != (3, 1):
        raise ValueError("Pair coordinates must have shape (3, 1)")
    mu = _as_scalar(mu, "mu", positive=True)

    # r points from the physical source to the target; r_image uses the image.
    r = (geometry.targets - geometry.sources)[:, 0]
    r_image = (geometry.targets - geometry.images)[:, 0]
    inverse_r = 1 / np.linalg.norm(r)
    inverse_image_r = 1 / np.linalg.norm(r_image)
    height = geometry.sources[2, 0]
    tensor = np.empty((3, 3))

    # i is the velocity component; j is the applied force component.
    for i in range(3):
        for j in range(3):
            same_component = i == j
            vertical_velocity = i == 2
            vertical_force = j == 2
            reflection_sign = 1 - 2 * vertical_force

            stokeslet = same_component * inverse_r + r[i] * r[j] * inverse_r**3
            image_stokeslet = (
                same_component * inverse_image_r
                + r_image[i] * r_image[j] * inverse_image_r**3
            )
            source_doublet = (
                same_component * inverse_image_r**3
                - 3 * r_image[i] * r_image[j] * inverse_image_r**5
            )
            stokes_doublet = (
                -vertical_velocity * r_image[j]
                + same_component * r_image[2]
                + r_image[i] * vertical_force
            ) * inverse_image_r**3
            stokes_doublet -= (
                3 * r_image[i] * r_image[2] * r_image[j] * inverse_image_r**5
            )

            # Blake (16): physical Stokeslet, image, source doublet, Stokes doublet.
            tensor[i, j] = (
                stokeslet
                - image_stokeslet
                + 2 * height**2 * reflection_sign * source_doublet
                - 2 * height * reflection_sign * stokes_doublet
            ) / (8 * np.pi * mu)

    if not np.isfinite(tensor).all():
        raise FloatingPointError("Blake tensor overflowed; rescale inputs")
    return tensor


def direct_velocity(geom=None, forces=None, *, mu=1.0, block_size=(256, 512)):
    """Apply the explicit Blake tensor in bounded blocks; no pair tensor is stored."""
    if geom is None and forces is None:
        geometry = prepare([[0, 0.2], [0, 0.3], [1, 0.7]], [[0.4], [0.2], [1.4]])
        forces = np.array([[1, 2], [3, -1], [2, 1]])
        reference = np.zeros((3, 1))
        for source_index in range(2):
            source = geometry.sources[:, source_index : source_index + 1]
            force = forces[:, source_index : source_index + 1]
            reference += blake_pair(source, geometry.targets) @ force
        value = direct_velocity(geometry, forces, block_size=(1, 1))
        np.testing.assert_allclose(value, reference, atol=1e-14)
        return None

    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    forces = _as_vector_array(forces, "forces")
    if forces.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _as_scalar(mu, "mu", positive=True)
    target_block_size, source_block_size = _block_sizes(block_size)
    velocity = np.zeros_like(geom.targets)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for target_start in range(0, velocity.shape[1], target_block_size):
            targets = slice(target_start, target_start + target_block_size)
            for source_start in range(0, forces.shape[1], source_block_size):
                sources = slice(source_start, source_start + source_block_size)

                # Work arrays have axes (component, target, source) within this block.
                r = geom.targets[:, targets, None] - geom.sources[:, None, sources]
                r_image = geom.targets[:, targets, None] - geom.images[:, None, sources]
                distance = np.linalg.norm(r, axis=0)
                image_distance = np.linalg.norm(r_image, axis=0)
                if np.any(distance == 0):
                    raise SingularInteractionError(
                        "A physical pair is unresolved after translation"
                    )
                inverse_r = 1 / distance
                inverse_image_r = 1 / image_distance
                block_forces = forces[:, sources]
                reflected_forces = block_forces.copy()
                reflected_forces[2] *= -1
                heights = geom.sources[2, sources]

                # Contract only the component axis, keeping each target/source pair.
                r_dot_force = np.einsum("its,is->ts", r, block_forces)
                image_r_dot_force = np.einsum("its,is->ts", r_image, block_forces)
                image_r_dot_reflected_force = np.einsum(
                    "its,is->ts", r_image, reflected_forces
                )
                for i in range(3):
                    component = (
                        block_forces[i] * inverse_r
                        + r[i] * r_dot_force * inverse_r**3
                        - block_forces[i] * inverse_image_r
                        - r_image[i] * image_r_dot_force * inverse_image_r**3
                    )
                    component += 2 * heights**2 * (
                        reflected_forces[i] * inverse_image_r**3
                        - 3 * r_image[i] * image_r_dot_reflected_force * inverse_image_r**5
                    )
                    stokes_doublet = (
                        reflected_forces[i] * r_image[2]
                        + r_image[i] * reflected_forces[2]
                    ) * inverse_image_r**3
                    stokes_doublet -= (
                        3 * r_image[i] * r_image[2]
                        * image_r_dot_reflected_force * inverse_image_r**5
                    )
                    if i == 2:
                        stokes_doublet -= image_r_dot_reflected_force * inverse_image_r**3
                    velocity[i, targets] += np.sum(
                        component - 2 * heights * stokes_doublet, axis=1
                    ) / (8 * np.pi * mu)

    if not np.isfinite(velocity).all():
        raise FloatingPointError("Direct velocity overflowed; rescale inputs")
    return velocity
