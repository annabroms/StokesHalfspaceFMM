"""Independent Blake (1971), equation (16); no Papkovich–Neuber assembly."""

import numpy as np

from .geometry import Geometry, SingularInteractionError, _array, _scalar, prepare


def _blocks(block_size):
    """Validate target/source block sizes before allocating interaction workspaces."""
    values = np.asarray(block_size)
    if values.shape != (2,) or values.dtype.kind not in "iu" or np.any(values <= 0):
        raise ValueError("block_size must contain two positive integers (targets, sources)")
    return tuple(int(v) for v in values)


def blake_pair(source=None, target=None, *, mu=1.0, wall_height=0.0):
    """Return the (3, 3) Blake velocity tensor for (3, 1) source and target arrays."""
    if source is None and target is None:
        y, x = np.array([[.1], [-.2], [.7]]), np.array([[.4], [.6], [1.2]])
        np.testing.assert_allclose(blake_pair(y, x), blake_pair(x, y).T, atol=1e-14)
        np.testing.assert_allclose(blake_pair(y, [[.4], [.6], [0]]), 0, atol=1e-14)
        return None
    g = prepare(source, target, wall_height)
    if g.sources.shape != (3, 1) or g.targets.shape != (3, 1):
        raise ValueError("Pair coordinates must have shape (3, 1)")
    mu = _scalar(mu, "mu", positive=True)
    r, R = (g.targets - g.sources)[:, 0], (g.targets - g.images)[:, 0]
    a, b, h = 1 / np.linalg.norm(r), 1 / np.linalg.norm(R), g.sources[2, 0]
    result = np.empty((3, 3))
    for i in range(3):
        for j in range(3):
            m = 1 - 2 * (j == 2)
            T = (i == j) * a + r[i] * r[j] * a**3
            TI = (i == j) * b + R[i] * R[j] * b**3
            D = (i == j) * b**3 - 3 * R[i] * R[j] * b**5
            E = (-(i == 2) * R[j] + (i == j) * R[2] + R[i] * (j == 2)) * b**3
            E -= 3 * R[i] * R[2] * R[j] * b**5
            result[i, j] = (T - TI + 2 * h**2 * m * D - 2 * h * m * E) / (8 * np.pi * mu)
    if not np.isfinite(result).all():
        raise FloatingPointError("Blake tensor overflowed; rescale inputs")
    return result


def direct_velocity(geom=None, forces=None, *, mu=1.0, block_size=(256, 512)):
    """Apply the explicit Blake tensor in bounded blocks; no pair tensor is stored."""
    if geom is None and forces is None:
        g = prepare([[0, .2], [0, .3], [1, .7]], [[.4], [.2], [1.4]])
        f = np.array([[1, 2], [3, -1], [2, 1]])
        reference = sum(blake_pair(g.sources[:, n:n+1], g.targets) @ f[:, n:n+1] for n in range(2))
        np.testing.assert_allclose(direct_velocity(g, f, block_size=(1, 1)), reference, atol=1e-14)
        return None
    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    f = _array(forces, "forces")
    if f.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _scalar(mu, "mu", positive=True)
    bt, bs = _blocks(block_size)
    u = np.zeros_like(geom.targets)
    with np.errstate(over="raise", divide="raise", invalid="raise"):
        for t0 in range(0, u.shape[1], bt):
            ti = slice(t0, t0 + bt)
            for s0 in range(0, f.shape[1], bs):
                si = slice(s0, s0 + bs)
                r = geom.targets[:, ti, None] - geom.sources[:, None, si]
                R = geom.targets[:, ti, None] - geom.images[:, None, si]
                rn, Rn = np.linalg.norm(r, axis=0), np.linalg.norm(R, axis=0)
                if np.any(rn == 0):
                    raise SingularInteractionError("A physical pair is unresolved after translation")
                a, b = 1 / rn, 1 / Rn
                ff = f[:, si]
                reflected = ff.copy()
                reflected[2] *= -1
                rf = np.einsum("its,is->ts", r, ff)
                Rf = np.einsum("its,is->ts", R, ff)
                Rg = np.einsum("its,is->ts", R, reflected)
                h = geom.sources[2, si]
                for i in range(3):
                    v = ff[i] * a + r[i] * rf * a**3 - ff[i] * b - R[i] * Rf * b**3
                    v += 2 * h**2 * (reflected[i] * b**3 - 3 * R[i] * Rg * b**5)
                    E = (reflected[i] * R[2] + R[i] * reflected[2]) * b**3
                    E -= 3 * R[i] * R[2] * Rg * b**5
                    if i == 2:
                        E -= Rg * b**3
                    u[i, ti] += np.sum(v - 2 * h * E, axis=1) / (8 * np.pi * mu)
    if not np.isfinite(u).all():
        raise FloatingPointError("Direct velocity overflowed; rescale inputs")
    return u
