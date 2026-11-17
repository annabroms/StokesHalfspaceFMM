"""Selection, independent tensor checks and mixed direct/FMM acceptance tests."""

import unittest
from unittest.mock import patch

import numpy as np

import blake
from blake.near_wall import select_targets, target_subset

try:
    import fmm3dpy
except ImportError:
    fmm3dpy = None


def fixture():
    """Make shallow sources and wall, near-wall, close bulk and distant targets."""
    rng = np.random.default_rng(384)
    sources = np.vstack((rng.random((2, 17)), 1e-6 * (1 + rng.random(17))))
    sources[:, -1] = [0.7, 0.6, 0.8]
    targets = np.column_stack((sources[:, :3], [0.2, 0.3, 1.0], sources[:, -1]))
    targets[2, :3] = [0, 1e-10, 1e-7]
    targets[0, -1] += 1e-7
    return sources, rng.normal(size=sources.shape), targets


class SelectionTests(unittest.TestCase):
    def test_physical_distance_override(self):
        """Use identical MATLAB/Python fixtures to check a cutoff in physical units."""
        sources = np.array([[0, .5, .7], [0, .2, .6], [1e-6, 2e-5, .8]])
        targets = np.array([[0, .5, .2, .70000005], [0, .2, .3, .6],
                            [0, 1e-10, 1, .8]])
        forces = np.array([[1, -.3, .7], [.2, .4, -.8], [-.1, .9, .5]])
        for scale in (1e-9, 1, 1e9):
            geom = blake.prepare(scale * sources, scale * targets)
            selected, radius = select_targets(geom, tol=1e-12, distance=scale * 1e-5)
            np.testing.assert_array_equal(selected, [True, False, False, True])
            self.assertAlmostEqual(radius, scale * 1e-5 / geom.span)
        with patch.dict("sys.modules", {"fmm3dpy": None}):
            value, info = blake.velocity(
                sources, forces, targets, correct_near_wall=True,
                near_wall_distance=10.0, return_info=True,
            )
        reference = blake.velocity(sources, forces, targets, method="blake")
        np.testing.assert_array_equal(value, reference)
        self.assertEqual(info["corrected_target_count"], 4)
        self.assertEqual(info["fmm_call_count"], 0)
        self.assertEqual(info["near_wall_distance"], 2 * blake.prepare(sources, targets).span)
        for bad in (0, -1, np.inf, "small", False, [1, 2]):
            with self.assertRaises(ValueError):
                blake.velocity(sources, forces, targets, correct_near_wall=True,
                               near_wall_distance=bad)

    def test_selection_and_scaling(self):
        """Check a bounded search against all-pair distances and scale invariance."""
        select_targets()
        y, f, x = fixture()
        for scale in (1e-9, 1, 1e9):
            geom = blake.prepare(scale * y, scale * x)
            for tol in (1e-6, 1e-9, 1e-12, 1e-300):
                selected, radius = select_targets(geom, tol=tol)
                # The dense comparison is deliberately confined to this tiny test.
                distance = np.linalg.norm(
                    geom.targets[:, :, None] - geom.sources[:, None, :], axis=0
                )
                expected = np.min(distance, axis=1) / geom.span <= radius
                np.testing.assert_array_equal(selected, expected)
            selected, _ = select_targets(geom)
            np.testing.assert_array_equal(selected, [True, True, True, False, True])
            subset = target_subset(geom, selected)
            np.testing.assert_array_equal(subset.targets, geom.targets[:, selected])
            self.assertIs(subset.sources, geom.sources)
            self.assertFalse(subset.targets.flags.writeable)
        for yy, xx in ((y[:, :0], x), (y, x[:, :0])):
            selected, _ = select_targets(blake.prepare(yy, xx))
            self.assertFalse(selected.any())

    def test_options_coincidences_and_direct_route(self):
        """Keep explicit coincidence errors and validate the optional boolean flag."""
        y, f, x = fixture()
        for bad in (1, 0, "true", None, [True]):
            with self.assertRaises(ValueError):
                blake.velocity(y, f, x, correct_near_wall=bad)
        with self.assertRaises(blake.SingularInteractionError):
            blake.velocity(y, 0 * f, y, correct_near_wall=True)
        direct = blake.velocity(y, f, x, method="blake")
        value, info = blake.velocity(
            y, f, x, method="blake", correct_near_wall=True, return_info=True
        )
        np.testing.assert_array_equal(value, direct)
        self.assertEqual(info["corrected_target_count"], 0)
        self.assertEqual(info["fmm_call_count"], 0)

    def test_all_direct_against_independent_pair_tensors(self):
        """Correct all selected targets without a backend, preserving physical units."""
        y, f, x = fixture()
        x = x[:, :3]
        reference = np.zeros_like(x)
        for m in range(x.shape[1]):
            for n in range(y.shape[1]):
                reference[:, m] += blake.blake_pair(y[:, n:n+1], x[:, m:m+1]) @ f[:, n]
        with patch.dict("sys.modules", {"fmm3dpy": None}):
            value, info = blake.velocity(
                y, f, x, tol=1e-12, correct_near_wall=True, return_info=True,
                block_size=(2, 3), mu=1.7,
            )
        np.testing.assert_allclose(1.7 * value, reference, rtol=2e-13, atol=2e-10)
        self.assertEqual(info["corrected_target_count"], 3)
        self.assertEqual(info["fmm_call_count"], 0)
        self.assertGreater(np.linalg.norm(value[:, 1:]), 1)
        # Below the FMM exclusion radius, a noncoincident pair still uses direct work.
        tiny_y = np.array([[0.0], [0.0], [1e-16]])
        tiny_x = np.array([[0.0, 1.0], [0.0, 0.0], [0.0, 1.0]])
        tiny_geom = blake.prepare(tiny_y, tiny_x)
        selected, _ = select_targets(tiny_geom)
        value = blake.apply(target_subset(tiny_geom, selected), np.ones((3, 1)),
                            correct_near_wall=True)
        np.testing.assert_array_equal(value, blake.direct_velocity(
            target_subset(tiny_geom, selected), np.ones((3, 1))))


@unittest.skipIf(fmm3dpy is None, "compiled fmm3dpy unavailable")
class CorrectedFMMTests(unittest.TestCase):
    def test_mixed_path_and_default(self):
        """Use exactly two FMM calls for safe targets and independent direct sums elsewhere."""
        y, f, x = fixture()
        baseline = blake.velocity(y, f, x)
        disabled = blake.velocity(y, f, x, correct_near_wall=False)
        np.testing.assert_array_equal(baseline, disabled)
        with patch.object(fmm3dpy, "stfmm3d", wraps=fmm3dpy.stfmm3d) as stokes, \
             patch.object(fmm3dpy, "lfmm3d", wraps=fmm3dpy.lfmm3d) as laplace:
            value, info = blake.velocity(y, f, x, correct_near_wall=True,
                                         return_info=True, tol=1e-12)
        self.assertEqual(info["corrected_target_count"], 4)
        self.assertEqual(info["fmm_call_count"], 2)
        self.assertEqual(stokes.call_count, 1)
        self.assertEqual(laplace.call_count, 1)
        self.assertEqual(stokes.call_args.kwargs["targets"].shape, (3, 1))
        self.assertEqual(laplace.call_args.kwargs["targets"].shape, (3, 1))
        direct = blake.velocity(y, f, x, method="blake")
        np.testing.assert_allclose(value, direct, rtol=2e-12, atol=2e-10)
        self.assertLess(np.linalg.norm(value[:, 0]), 1e-9)
        self.assertGreater(np.linalg.norm(value[:, 1]), 1e-3)
        with self.assertRaises(ImportError), patch.dict("sys.modules", {"fmm3dpy": None}):
            blake.velocity(y, f, x, correct_near_wall=True)

    def test_translation_linearity_and_reciprocity(self):
        """Check mixed routing under shifts, viscosity changes and reciprocal work."""
        y, f, x = fixture()
        options = dict(correct_near_wall=True, tol=1e-12)
        value = blake.velocity(y, f, x, **options)
        np.testing.assert_allclose(blake.velocity(y, 2*f, x, **options), 2*value,
                                   rtol=1e-12, atol=1e-10)
        shift = np.array([[7.0], [-4.0], [1.3]])
        shifted_y, shifted_x = y + shift, x + shift
        shifted = blake.velocity(shifted_y, f, shifted_x, wall_height=1.3,
                                 mu=1.7, **options)
        # Compare with direct at the same rounded input coordinates.
        reference = blake.velocity(shifted_y, f, shifted_x, wall_height=1.3,
                                   mu=1.7, method="blake")
        np.testing.assert_allclose(shifted, reference, rtol=1e-10, atol=1e-9)

        # Include a close pair and a distant pair, with both clouds above the wall.
        a = y[:, [0, -1]].copy()
        b = a.copy()
        b[0] += [1e-7, .1]
        fa = np.array([[1., .3], [-.4, .7], [.2, -.6]])
        fb = np.array([[.5, -.2], [.1, -.3], [-.8, 1.]])
        ab = blake.velocity(a, fa, b, **options)
        ba = blake.velocity(b, fb, a, **options)
        left = np.sum(fb * ab)
        right = np.sum(fa * ba)
        scale = np.sum(np.abs(fb * ab)) + np.sum(np.abs(fa * ba))
        self.assertLess(abs(left - right) / scale, 2e-12)


if __name__ == "__main__":
    unittest.main()
