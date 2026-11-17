"""Regression checks for the default compiled proximity guard."""

import unittest

import numpy as np

from blake._proximity import unresolved_pairs
from blake.geometry import FMMResolutionError, guard, prepare


class ProximityTests(unittest.TestCase):
    """Preserve inclusive radii, scale invariance and first-target errors."""

    def test_no_argument_self_test(self):
        """Exercise the compiled index and parallel query independently."""
        self.assertIsNone(unresolved_pairs())

    def test_inclusive_radius_and_scale(self):
        """Reject equal/below-threshold pairs and accept the next float above it."""
        tau = 64*np.finfo(float).eps
        sources = np.array([[0., -1., 1.], [0., -1., 1.], [.5, 1., 1.]])
        for scale in (1e-200, 1e-9, 1., 1e9, 1e200):
            for dx in (np.nextafter(2*tau, 0.), 2*tau, np.nextafter(2*tau, np.inf)):
                with self.subTest(scale=scale, dx=dx):
                    geom = prepare(sources*scale, np.array([[dx], [0.], [.5]])*scale)
                    source = np.concatenate((geom.sources, geom.images), axis=1)/geom.span
                    delta = (source-geom.targets[:, :1]/geom.span)/tau
                    expected = np.any(np.sum(delta*delta, axis=0) <= 1.)
                    if expected:
                        with self.assertRaises(FMMResolutionError):
                            guard(geom)
                    else:
                        self.assertIsNone(guard(geom))

    def test_dense_chains_and_first_target(self):
        """Retain all clustered candidates and report the first offending target."""
        tau = 64*np.finfo(float).eps
        sources = np.repeat(np.array([[0.], [0.], [.5]]), 1025, axis=1)
        sources[0] += np.arange(1025) % 7 * tau/20
        sources = np.column_stack((sources, [-1., -1., 1.], [1., 1., 1.]))
        targets = np.array([[.4, tau, 0.], [.3, 0., tau], [.8, .5, .5]])
        geom = prepare(sources, targets)
        with self.assertRaisesRegex(FMMResolutionError, r"Target 1 .*physical source"):
            guard(geom)
        self.assertFalse(geom.sources.flags.writeable)
        self.assertFalse(geom.targets.flags.writeable)


if __name__ == "__main__":
    unittest.main()
