"""Deterministic direct, FMM and physical-identity acceptance tests."""

import unittest
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

import blake
from blake.fmm import fmm_velocity
from blake.geometry import guard

try:
    import fmm3dpy
except ImportError:
    fmm3dpy = None


def fixture(ns=13, nt=19, seed=48):
    """Make independent point sets with unequal counts and deterministic signed forces."""
    rng = np.random.default_rng(seed)
    y = np.vstack((rng.random((2, ns)), .2 + rng.random(ns)))
    x = np.vstack((rng.random((2, nt)), .1 + rng.random(nt)))
    return y, rng.standard_normal((3, ns)), x


class DirectTests(unittest.TestCase):
    def test_no_argument_self_tests(self):
        """Run every dependency-free computational self-test independently."""
        for function in (blake.prepare, guard, blake.blake_pair, blake.direct_velocity,
                         blake.apply, blake.velocity, blake.constituent_scale, blake.metrics):
            function()

    def test_blocked_sum_against_pair_tensors(self):
        """Check independent pair entries against contractions with several block sizes."""
        y, f, x = fixture()
        y[:, -1] = y[:, 0]
        reference = np.zeros_like(x)
        for m in range(x.shape[1]):
            for n in range(y.shape[1]):
                reference[:, m] += blake.blake_pair(y[:, n:n+1], x[:, m:m+1]) @ f[:, n]
        for block_size in ((1, 1), (3, 4), (256, 512)):
            u = blake.velocity(y, f, x, method="blake", block_size=block_size)
            np.testing.assert_allclose(u, reference, rtol=2e-13, atol=2e-14)
        g = blake.prepare(y, x)
        np.testing.assert_allclose(blake.apply(g, f, method="blake"), reference, atol=2e-14)

    def test_pair_wall_and_reciprocity(self):
        """Verify all tensor components for translated walls and several source heights."""
        for h in (1e-6, .7, 1e3):
            y, x = np.array([[.2], [-.3], [h]]), np.array([[.8], [.5], [1.1]])
            for wall in (0, 1.3):
                shift = np.array([[0], [0], [wall]])
                B = blake.blake_pair(y + shift, x + shift, wall_height=wall, mu=1.7)
                C = blake.blake_pair(x + shift, y + shift, wall_height=wall, mu=1.7)
                np.testing.assert_allclose(B, C.T, rtol=1e-11, atol=1e-14)
                w = x + shift
                w[2] = wall
                np.testing.assert_allclose(blake.blake_pair(y + shift, w, wall_height=wall), 0, atol=1e-13)

    def test_linearity_translation_viscosity_scaling(self):
        """Verify physical units, coordinate scaling and independent force superposition."""
        y, f, x = fixture()
        extra = np.roll(f, 1, axis=0)
        u = blake.velocity(y, f, x, method="blake")
        v = blake.velocity(y, extra, x, method="blake")
        w = blake.velocity(y, 2*f - 3*extra, x, method="blake")
        np.testing.assert_allclose(w, 2*u - 3*v, rtol=1e-12, atol=1e-13)
        shift = np.array([[7], [-4], [1.3]])
        w = blake.velocity(y + shift, f, x + shift, method="blake", mu=2.5, wall_height=1.3)
        np.testing.assert_allclose(2.5*w, u, rtol=2e-12, atol=1e-13)
        for scale in (1e-9, 1e-3, 1e3, 1e9):
            w = blake.velocity(scale*y, f, scale*x, method="blake")
            np.testing.assert_allclose(scale*w, u, rtol=3e-12, atol=1e-13)

    def test_input_coincidence_empty_and_duplicates(self):
        """Reject singular or invalid inputs and preserve empty/singleton array contracts."""
        y, f, x = fixture(1, 1)
        for method in ("blake", "fmm"):
            with self.assertRaises(blake.SingularInteractionError):
                blake.velocity(y, np.zeros_like(f), y, method=method)
            self.assertEqual(blake.velocity(y, f, x[:, :0], method=method).shape, (3, 0))
            np.testing.assert_array_equal(blake.velocity(y[:, :0], f[:, :0], x, method=method), np.zeros((3, 1)))
        for bad in (y.T, np.ones((3, 1), dtype=complex), np.full((3, 1), np.nan)):
            with self.assertRaises(ValueError):
                blake.prepare(bad, x)
        for bad in (0, -1, np.inf, 1j):
            with self.assertRaises(ValueError):
                blake.velocity(y, f, x, mu=bad)
        for options in ({"tol": 0}, {"tol": 1}, {"method": "typo"}, {"block_size": (0, 1)}):
            with self.assertRaises(ValueError):
                blake.velocity(y, f, x, **options)
        for yy, xx in (([[0], [0], [0]], x), (y, [[0], [0], [-1]])):
            with self.assertRaises(ValueError):
                blake.prepare(yy, xx)
        with self.assertRaises(ValueError):
            blake.velocity(y, np.ones((3, 2)), x)
        u = blake.velocity(y, f, x, method="blake")
        duplicate = blake.velocity(np.repeat(y, 2, axis=1), np.repeat(f, 2, axis=1), x, method="blake")
        np.testing.assert_allclose(duplicate, 2*u)

    def test_fmm_resolution_guard_and_direct_escape(self):
        """Reject physical and reflected near coincidences while retaining direct evaluation."""
        fixtures = [
            (np.array([[0, 1], [0, 1], [1e-16, 1]]), np.zeros((3, 1))),
            (np.array([[0, 1], [0, 1], [1, 1]]), np.array([[1e-16], [0], [1]])),
        ]
        for y, x in fixtures:
            with self.assertRaises(blake.FMMResolutionError):
                blake.velocity(y, np.ones_like(y), x)
            self.assertTrue(np.isfinite(blake.velocity(y, np.ones_like(y), x, method="blake")).all())

    def test_metrics_and_blocked_diagnostics(self):
        """Keep relative wall error undefined and constituent scaling block-independent."""
        y, f, x = fixture()
        g = blake.prepare(y, x)
        a = blake.constituent_scale(g, f, block_size=(2, 3))
        b = blake.constituent_scale(g, f)
        np.testing.assert_allclose(a, b, atol=1e-13)
        u = blake.apply(g, f, method="blake")
        self.assertTrue(np.all(np.linalg.norm(u, axis=0) <= a))
        out = blake.metrics(np.ones((3, 1)), np.zeros((3, 1)), np.ones(1))
        self.assertTrue(np.isnan(out["global_relative"]))
        self.assertEqual(out["max_absolute"], np.sqrt(3))
        self.assertTrue(np.isinf(blake.metrics(np.ones((3, 1)), np.zeros((3, 1)), np.zeros(1))["constituent_normalized"]))

    def test_dependency_absence_is_explicit(self):
        """A missing backend must raise instead of selecting direct evaluation silently."""
        y, f, x = fixture(1, 1)
        with patch.dict("sys.modules", {"fmm3dpy": None}):
            with self.assertRaises(ImportError):
                blake.velocity(y, f, x)
            self.assertTrue(np.isfinite(blake.velocity(y, f, x, method="blake")).all())


@unittest.skipIf(fmm3dpy is None, "INCOMPLETE: fmm3dpy unavailable")
class FMMTests(unittest.TestCase):
    def test_no_argument_fmm_self_test(self):
        """Exercise the actual compiled Stokes and Laplace wrappers."""
        fmm_velocity()

    def test_normalization_probes(self):
        """Check free-space velocity and charge/dipole potentials and gradients separately."""
        from blake.validation import validate_wrappers
        report = validate_wrappers()
        for values in report.values():
            self.assertLess(values, 1e-10)

    def test_accuracy_wall_and_reciprocity(self):
        """Run all tolerance-sweep fixtures including distributed multipole reciprocity."""
        from blake.validation import validate_velocity
        report = validate_velocity()
        self.assertEqual(len(report["cases"]), 12)

    def test_call_contract_and_failures(self):
        """Check exactly two target-only calls and propagate statuses and malformed outputs."""
        y, f, x = fixture()
        with patch.object(fmm3dpy, "stfmm3d", wraps=fmm3dpy.stfmm3d) as st, patch.object(fmm3dpy, "lfmm3d", wraps=fmm3dpy.lfmm3d) as lf:
            _, info = blake.velocity(y, f, x, return_info=True)
            self.assertEqual((st.call_count, lf.call_count, info["fmm_call_count"]), (1, 1, 2))
            self.assertEqual(st.call_args.kwargs["ifppreg"], 0)
            self.assertEqual(st.call_args.kwargs["ifppregtarg"], 1)
            self.assertEqual(lf.call_args.kwargs["pg"], 0)
            self.assertEqual(lf.call_args.kwargs["pgt"], 2)
            np.testing.assert_array_equal(lf.call_args.kwargs["charges"], -f[2])
            self.assertLessEqual(info["stokes_seconds"] + info["laplace_seconds"], info["total_seconds"])
        for backend, fake in (
            ("stfmm3d", SimpleNamespace(ier=4)),
            ("lfmm3d", SimpleNamespace(ier=8)),
            ("stfmm3d", SimpleNamespace(ier=0, pottarg=np.zeros((1, x.size)))),
            ("stfmm3d", SimpleNamespace(ier=0, pottarg=np.full_like(x, np.nan))),
            ("lfmm3d", SimpleNamespace(ier=0, pottarg=None)),
        ):
            with patch.object(fmm3dpy, backend, return_value=fake):
                with self.assertRaises(RuntimeError):
                    blake.velocity(y, f, x)

    def test_fmm_units_scaling_duplicates_and_linearity(self):
        """Verify FMM physical conventions independently of the direct comparison fixtures."""
        y, f, x = fixture(41, 23)
        u = blake.velocity(y, f, x, tol=1e-12)
        shift = np.array([[7], [-4], [1.3]])
        v = blake.velocity(y + shift, f, x + shift, mu=2.5, wall_height=1.3, tol=1e-12)
        np.testing.assert_allclose(2.5*v, u, rtol=1e-8, atol=1e-10)
        for scale in (1e-9, 1e-3, 1e3, 1e9):
            v = blake.velocity(scale*y, f, scale*x, tol=1e-12)
            np.testing.assert_allclose(scale*v, u, rtol=1e-8, atol=1e-10)
        v = blake.velocity(np.repeat(y, 2, axis=1), np.repeat(f, 2, axis=1), x, tol=1e-12)
        np.testing.assert_allclose(v, 2*u, rtol=1e-8, atol=1e-10)
        other = np.roll(f, 1, axis=0)
        v = blake.velocity(y, other, x, tol=1e-12)
        w = blake.velocity(y, 2*f - 3*other, x, tol=1e-12)
        np.testing.assert_allclose(w, 2*u - 3*v, rtol=1e-8, atol=1e-10)
        self.assertEqual(blake.velocity(y[:, :1], f[:, :1], x[:, :1]).shape, (3, 1))


if __name__ == "__main__":
    unittest.main()
