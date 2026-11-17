"""Opt-in blocked constituent scales and cancellation-aware velocity metrics."""

import numpy as np

from .direct import _blocks
from .geometry import Geometry, _array, _scalar, prepare


def constituent_scale(geom=None, forces=None, *, mu=1.0, block_size=(256, 512)):
    """Sum magnitudes before cancellation; O(Ns*Nt) time with bounded workspace."""
    if geom is None and forces is None:
        g = prepare([[0], [0], [1]], [[1], [0], [0]])
        scale = constituent_scale(g, [[1], [2], [3]])
        assert scale.shape == (1,) and scale[0] > 0
        np.testing.assert_array_equal(constituent_scale(g, np.zeros((3, 1))), [0])
        return None
    if not isinstance(geom, Geometry):
        raise ValueError("Use geometry returned by prepare()")
    f = _array(forces, "forces")
    if f.shape != geom.sources.shape:
        raise ValueError("forces must have the same shape as sources")
    mu = _scalar(mu, "mu", True)
    bt, bs = _blocks(block_size)
    result = np.zeros(geom.targets.shape[1])
    for t0 in range(0, result.size, bt):
        ti = slice(t0, t0 + bt)
        for s0 in range(0, f.shape[1], bs):
            si = slice(s0, s0 + bs)
            ff = f[:, None, si]
            fs = ff.copy()
            fs[2] *= -1
            r = geom.targets[:, ti, None] - geom.sources[:, None, si]
            R = geom.targets[:, ti, None] - geom.images[:, None, si]
            a, b = 1 / np.linalg.norm(r, axis=0), 1 / np.linalg.norm(R, axis=0)
            v = (ff * a + r * np.sum(r * ff, axis=0) * a**3) / (8 * np.pi)
            vi = (fs * b + R * np.sum(R * fs, axis=0) * b**3) / (8 * np.pi)
            q, d = fs[2], fs * geom.sources[2, si]
            Rd = np.sum(R * d, axis=0)
            phi = (q * b + Rd * b**3) / (4 * np.pi)
            grad = (-q * R * b**3 + d * b**3 - 3 * R * Rd * b**5) / (4 * np.pi)
            scale = np.linalg.norm(v, axis=0) + np.linalg.norm(vi, axis=0)
            scale += geom.targets[2, ti, None] * np.linalg.norm(grad, axis=0) + np.abs(phi)
            result[ti] += np.sum(scale, axis=1) / mu
    if not np.isfinite(result).all():
        raise FloatingPointError("Constituent scale overflowed; rescale inputs")
    return result


def metrics(value=None, reference=None, scale=None):
    """Report absolute, global relative and constituent-normalized velocity errors."""
    if value is None and reference is None and scale is None:
        out = metrics(np.zeros((3, 1)), np.zeros((3, 1)), np.ones(1))
        assert np.isnan(out["global_relative"]) and out["max_absolute"] == 0
        return None
    value, reference = _array(value, "value"), _array(reference, "reference")
    scale = np.asarray(scale, dtype=float)
    if value.shape != reference.shape or scale.shape != (value.shape[1],):
        raise ValueError("Expected matching (3, Nt) fields and (Nt,) scale")
    if not np.isfinite(scale).all() or np.any(scale < 0):
        raise ValueError("scale must be finite and nonnegative")
    pointwise = np.linalg.norm(value - reference, axis=0)
    ratio = np.zeros_like(pointwise)
    np.divide(pointwise, scale, out=ratio, where=scale > 0)
    ratio[(scale == 0) & (pointwise > 0)] = np.inf
    norm = np.linalg.norm(reference)
    relative = float(np.linalg.norm(value - reference) / norm) if norm > 100 * np.finfo(float).eps * np.linalg.norm(scale) else float("nan")
    return dict(max_absolute=float(pointwise.max(initial=0)), global_relative=relative,
                constituent_normalized=float(ratio.max(initial=0)))
