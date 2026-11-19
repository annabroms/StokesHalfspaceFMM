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


def blake_pair(source=None, target=None, *, mu=1.0, wall_height=0.0, derivatives=False):
    """Return the (3, 3) Blake velocity tensor for (3, 1) source and target arrays.

    derivatives=True also returns pressure P[j] and G[i, j, k] = dB[i, j]/dx_k.
    """
    if source is None and target is None:
        source = np.array([[0.1], [-0.2], [0.7]])
        target = np.array([[0.4], [0.6], [1.2]])
        forward = blake_pair(source, target)
        reverse = blake_pair(target, source)
        np.testing.assert_allclose(forward, reverse.T, atol=1e-14)
        wall_tensor = blake_pair(source, [[0.4], [0.6], [0]])
        np.testing.assert_allclose(wall_tensor, 0, atol=1e-14)
        tensor, _, gradient = blake_pair(source, target, mu=1.7, derivatives=True)
        np.testing.assert_array_equal(tensor, blake_pair(source, target, mu=1.7))
        np.testing.assert_allclose(np.einsum("iji->j", gradient), 0, atol=1e-14)
        _, _, wall_gradient = blake_pair(source, [[0.4], [0.6], [0]], derivatives=True)
        np.testing.assert_allclose(wall_gradient[:, :, :2], 0, atol=1e-14)
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
    if not derivatives:
        return tensor

    # Explicit analytic target derivatives of each Blake (16) term; P is independent of mu.
    R, r_vec = r_image, r
    a, b = inverse_r, inverse_image_r
    pressure = np.empty(3)
    gradient = np.empty((3, 3, 3))
    for j in range(3):
        m = 1 - 2 * (j == 2)
        pressure[j] = (
            r_vec[j] * a**3 - R[j] * b**3
            - 2 * height * m * ((j == 2) * b**3 - 3 * R[2] * R[j] * b**5)
        ) / (4 * np.pi)
        for i in range(3):
            for k in range(3):
                d_stokeslet = (
                    -(i == j) * r_vec[k] * a**3
                    + ((i == k) * r_vec[j] + r_vec[i] * (j == k)) * a**3
                    - 3 * r_vec[i] * r_vec[j] * r_vec[k] * a**5
                )
                d_image = (
                    -(i == j) * R[k] * b**3
                    + ((i == k) * R[j] + R[i] * (j == k)) * b**3
                    - 3 * R[i] * R[j] * R[k] * b**5
                )
                d_source_doublet = (
                    -3 * ((i == j) * R[k] + (i == k) * R[j] + R[i] * (j == k)) * b**5
                    + 15 * R[i] * R[j] * R[k] * b**7
                )
                d_stokes_doublet = (
                    -(i == 2) * (j == k) * b**3
                    + 3 * (i == 2) * R[j] * R[k] * b**5
                    + ((i == j) * (k == 2) + (i == k) * (j == 2)) * b**3
                    - 3 * ((i == j) * R[2] + R[i] * (j == 2)) * R[k] * b**5
                    - 3 * ((i == k) * R[2] * R[j] + R[i] * (k == 2) * R[j]
                           + R[i] * R[2] * (j == k)) * b**5
                    + 15 * R[i] * R[2] * R[j] * R[k] * b**7
                )
                gradient[i, j, k] = (
                    d_stokeslet - d_image
                    + 2 * height**2 * m * d_source_doublet
                    - 2 * height * m * d_stokes_doublet
                ) / (8 * np.pi * mu)
    if not (np.isfinite(pressure).all() and np.isfinite(gradient).all()):
        raise FloatingPointError("Blake derivatives overflowed; rescale inputs")
    return tensor, pressure, gradient


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


def direct_fields(geom=None, forces=None, *, order=3, mu=1.0, block_size=(256, 512)):
    """Blocked Blake velocity, pressure (order>=2) and grad_u[i, j, m]=du_i/dx_j (order 3).

    The derivative contraction uses Blake's tensor only, never the image representation.
    """
    if geom is None and forces is None:
        geometry = prepare([[0, 0.2], [0, 0.3], [1, 0.7]], [[0.4, 0.1], [0.2, 0.5], [1.4, 0]])
        forces = np.array([[1, 2], [3, -1], [2, 1]])
        value = direct_fields(geometry, forces, mu=1.3, block_size=(1, 1))
        pressure, gradient = np.zeros(2), np.zeros((3, 3, 2))
        for m in range(2):
            for n in range(2):
                _, p, g = blake_pair(geometry.sources[:, n:n+1], geometry.targets[:, m:m+1],
                                     mu=1.3, derivatives=True)
                pressure[m] += p @ forces[:, n]
                gradient[:, :, m] += np.einsum("ijk,j->ik", g, forces[:, n])
        np.testing.assert_allclose(value["pressure"], pressure, atol=1e-14)
        np.testing.assert_allclose(value["grad_u"], gradient, atol=1e-14)
        return None

    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    if order not in (1, 2, 3):
        raise ValueError("order must be 1 (velocity), 2 (pressure) or 3 (gradient)")
    forces = _as_vector_array(forces, "forces")
    if forces.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _as_scalar(mu, "mu", positive=True)
    target_block_size, source_block_size = _block_sizes(block_size)
    nt = geom.targets.shape[1]
    fields = {"velocity": direct_velocity(geom, forces, mu=mu, block_size=block_size)}
    if order < 2:
        return fields
    pressure = np.zeros(nt)
    gradient = np.zeros((3, 3, nt)) if order >= 3 else None
    identity = np.eye(3)

    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for target_start in range(0, nt, target_block_size):
            targets = slice(target_start, target_start + target_block_size)
            for source_start in range(0, forces.shape[1], source_block_size):
                sources = slice(source_start, source_start + source_block_size)

                # Axes (component, target, source); velocity already rejected coincidences.
                r = geom.targets[:, targets, None] - geom.sources[:, None, sources]
                R = geom.targets[:, targets, None] - geom.images[:, None, sources]
                a = 1 / np.linalg.norm(r, axis=0)
                b = 1 / np.linalg.norm(R, axis=0)
                a3, b3 = a**3, b**3
                a5, b5, b7 = a3 * a * a, b3 * b * b, b3 * b**4
                f = forces[:, sources]
                g = f.copy()
                g[2] *= -1
                h = geom.sources[2, sources]
                rf = np.einsum("its,is->ts", r, f)
                Rf = np.einsum("its,is->ts", R, f)
                Rg = np.einsum("its,is->ts", R, g)

                # Blake pressure P_j contracted with f_j: m_j f_j = g_j.
                pressure[targets] += np.sum(
                    rf * a3 - Rf * b3 - 2 * h * (g[2] * b3 - 3 * R[2] * Rg * b5), axis=1
                ) / (4 * np.pi)
                if gradient is None:
                    continue
                # G(i,j,k) contracted with f_j (Stokeslets) or g_j (reflected terms).
                for i in range(3):
                    for k in range(3):
                        d_stokeslet = (
                            -f[i] * r[k] * a3 + identity[i, k] * rf * a3
                            + r[i] * f[k] * a3 - 3 * r[i] * rf * r[k] * a5
                        )
                        d_image = (
                            -f[i] * R[k] * b3 + identity[i, k] * Rf * b3
                            + R[i] * f[k] * b3 - 3 * R[i] * Rf * R[k] * b5
                        )
                        d_source_doublet = (
                            -3 * (g[i] * R[k] + identity[i, k] * Rg + R[i] * g[k]) * b5
                            + 15 * R[i] * Rg * R[k] * b7
                        )
                        d_stokes_doublet = (
                            -identity[i, 2] * g[k] * b3
                            + 3 * identity[i, 2] * Rg * R[k] * b5
                            + (g[i] * identity[k, 2] + identity[i, k] * g[2]) * b3
                            - 3 * (g[i] * R[2] + R[i] * g[2]) * R[k] * b5
                            - 3 * (identity[i, k] * R[2] * Rg + R[i] * identity[k, 2] * Rg
                                   + R[i] * R[2] * g[k]) * b5
                            + 15 * R[i] * R[2] * Rg * R[k] * b7
                        )
                        gradient[i, k, targets] += np.sum(
                            d_stokeslet - d_image + 2 * h**2 * d_source_doublet
                            - 2 * h * d_stokes_doublet, axis=1
                        ) / (8 * np.pi * mu)

    fields["pressure"] = pressure
    if gradient is not None:
        fields["grad_u"] = gradient
    if not all(np.isfinite(value).all() for value in fields.values()):
        raise FloatingPointError("Direct fields overflowed; rescale inputs")
    return fields
