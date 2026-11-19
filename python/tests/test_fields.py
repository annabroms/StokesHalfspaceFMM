"""Pressure, velocity-gradient and traction acceptance tests (direct and FMM)."""

import unittest
from unittest.mock import patch

import numpy as np

import blake
from blake.fmm import fmm_fields

try:
    import fmm3dpy
except ImportError:
    fmm3dpy = None


def fixture(ns=9, nt=11, seed=51):
    """Make sources, signed forces and targets including two on the wall."""
    rng = np.random.default_rng(seed)
    y = np.vstack((rng.random((2, ns)), .3 + rng.random(ns)))
    x = np.vstack((rng.random((2, nt)), np.r_[0, 0, .01, .1 + rng.random(nt - 3)]))
    return y, rng.standard_normal((3, ns)), x


def unit_normals(count, seed=52):
    """Draw deterministic random unit normals."""
    n = np.random.default_rng(seed).standard_normal((3, count))
    return n / np.linalg.norm(n, axis=0)


def pair_fields(y, f, x, mu):
    """Sum the readable pair pressure and derivative tensors over all pairs."""
    pressure, gradient = np.zeros(x.shape[1]), np.zeros((3, 3, x.shape[1]))
    for m in range(x.shape[1]):
        for n in range(y.shape[1]):
            _, p, g = blake.blake_pair(y[:, n:n+1], x[:, m:m+1], mu=mu, derivatives=True)
            pressure[m] += p @ f[:, n]
            gradient[:, :, m] += np.einsum("ijk,j->ik", g, f[:, n])
    return pressure, gradient


class DirectFieldTests(unittest.TestCase):
    def test_no_argument_self_tests(self):
        """Run every dependency-free field self-test independently."""
        for function in (blake.blake_pair, blake.direct_fields, blake.apply_fields, blake.fields):
            function()

    def test_blocked_fields_against_pair_derivatives(self):
        """Check the blocked contraction against independent pair entries for several blocks."""
        y, f, x = fixture()
        pressure, gradient = pair_fields(y, f, x, 2.3)
        for block_size in ((1, 1), (3, 4), (256, 512)):
            value = blake.fields(y, f, x, requested="gradient", method="blake",
                                 mu=2.3, block_size=block_size)
            np.testing.assert_allclose(value["pressure"], pressure, rtol=1e-12, atol=1e-14)
            np.testing.assert_allclose(value["grad_u"], gradient, rtol=1e-12, atol=1e-14)
            np.testing.assert_array_equal(
                value["velocity"], blake.velocity(y, f, x, method="blake", mu=2.3,
                                                  block_size=block_size))

    def test_finite_difference_step_study(self):
        """Finite differences are test diagnostics only: grad_u and Stokes momentum balance."""
        y, f = np.array([[0], [0], [.7]]), np.array([[1.], [-2], [3]])
        mu = 1.4
        for target in ([.4, .3, 1.2], [.1, -.2, .05]):
            x = np.array(target)[:, None]
            exact = blake.fields(y, f, x, requested="gradient", method="blake", mu=mu)
            errors = []
            for step in (1e-2, 1e-3, 1e-4):
                fd = np.zeros((3, 3))
                for j in range(3):
                    d = np.zeros((3, 1))
                    d[j] = step
                    fd[:, j] = (blake.velocity(y, f, x + d, method="blake", mu=mu)
                                - blake.velocity(y, f, x - d, method="blake", mu=mu))[:, 0] / (2 * step)
                errors.append(np.linalg.norm(fd - exact["grad_u"][:, :, 0])
                              / np.linalg.norm(exact["grad_u"]))
            # Central differences converge at second order (measured ratio about 100).
            self.assertLess(errors[1], errors[0] / 80)
            self.assertLess(errors[2], errors[1] / 80)
            self.assertLess(errors[2], 1e-6)

            # grad(p) = mu * laplacian(u) away from the source, from central differences.
            step = 1e-4
            grad_p, laplacian = np.zeros(3), np.zeros(3)
            for j in range(3):
                d = np.zeros((3, 1))
                d[j] = step
                plus = blake.fields(y, f, x + d, requested="gradient", method="blake", mu=mu)
                minus = blake.fields(y, f, x - d, requested="gradient", method="blake", mu=mu)
                grad_p[j] = (plus["pressure"][0] - minus["pressure"][0]) / (2 * step)
                laplacian += (plus["grad_u"][:, j, 0] - minus["grad_u"][:, j, 0]) / (2 * step)
            np.testing.assert_allclose(grad_p, mu * laplacian, rtol=1e-6,
                                       atol=1e-7 * np.linalg.norm(grad_p))

    def test_divergence_wall_and_traction(self):
        """Check incompressibility, no-slip derivatives at the wall and traction assembly."""
        y, f, x = fixture()
        normals = unit_normals(x.shape[1])
        value = blake.fields(y, f, x, requested=("traction",), normals=normals,
                             method="blake", mu=1.9)
        scale = np.linalg.norm(value["grad_u"], axis=(0, 1))
        np.testing.assert_allclose(np.trace(value["grad_u"]), 0, atol=1e-13 * scale.max())
        # On the wall u=0 for all x, y: tangential derivatives vanish, so dw/dz does too.
        wall = value["grad_u"][:, :, :2]
        np.testing.assert_allclose(wall[:, :2], 0, atol=1e-14 * scale.max())
        np.testing.assert_allclose(wall[2, 2], 0, atol=1e-14 * scale.max())
        self.assertTrue(np.all(np.abs(wall[:2, 2]) > 0))
        for m in range(x.shape[1]):
            G = value["grad_u"][:, :, m]
            stress = -value["pressure"][m] * np.eye(3) + 1.9 * (G + G.T)
            np.testing.assert_allclose(value["traction"][:, m], stress @ normals[:, m],
                                       rtol=1e-13, atol=1e-15)

    def test_viscosity_and_length_scaling(self):
        """Pressure and traction ignore mu; gradients scale as 1/mu and 1/length^2."""
        y, f, x = fixture()
        normals = unit_normals(x.shape[1])
        base = blake.fields(y, f, x, requested="traction", normals=normals, method="blake")
        viscous = blake.fields(y, f, x, requested="traction", normals=normals,
                               method="blake", mu=3.5)
        np.testing.assert_allclose(viscous["pressure"], base["pressure"], rtol=1e-13)
        np.testing.assert_allclose(viscous["traction"], base["traction"], rtol=1e-12, atol=1e-15)
        np.testing.assert_allclose(viscous["grad_u"], base["grad_u"] / 3.5, rtol=1e-13, atol=1e-16)
        for s in (1e-3, 1e3):
            scaled = blake.fields(s * y, f, s * x, requested="gradient", method="blake",
                                  wall_height=0)
            np.testing.assert_allclose(scaled["pressure"], base["pressure"] / s**2,
                                       rtol=1e-11, atol=1e-13 / s**2)
            np.testing.assert_allclose(scaled["grad_u"], base["grad_u"] / s**2,
                                       rtol=1e-11, atol=1e-13 / s**2)

    def test_translated_wall(self):
        """A shifted wall and coordinates give the same fields."""
        y, f, x = fixture()
        base = blake.fields(y, f, x, requested="gradient", method="blake")
        shift = np.array([[.3], [-.2], [1.3]])
        moved = blake.fields(y + shift, f, x + shift, requested="gradient",
                             method="blake", wall_height=1.3)
        for name in ("pressure", "grad_u"):
            np.testing.assert_allclose(moved[name], base[name], rtol=1e-11, atol=1e-13)

    def test_options_shapes_and_errors(self):
        """Reject ambiguous requests and normals; preserve empty output shapes."""
        y, f, x = fixture(2, 3)
        only_pressure = blake.fields(y, f, x, requested="pressure", method="blake")
        self.assertEqual(set(only_pressure), {"velocity", "pressure"})
        self.assertEqual(only_pressure["pressure"].shape, (3,))
        for requested in ((), ("vorticity",), 3, ("pressure", 2)):
            with self.assertRaises(ValueError):
                blake.fields(y, f, x, requested=requested, method="blake")
        for normals in (None, np.zeros((3, 3)), np.ones((3, 3)), np.eye(3)[:, :2],
                        np.full((3, 3), np.nan)):
            with self.assertRaises(ValueError):
                blake.fields(y, f, x, requested="traction", normals=normals, method="blake")
        with self.assertRaises(ValueError):
            blake.fields(y, f, x, requested="pressure", normals=np.eye(3), method="blake")
        with self.assertRaises(ValueError):
            blake.direct_fields(blake.prepare(y, x), f, order=4)
        with self.assertRaises(blake.SingularInteractionError):
            blake.fields(y, f, y[:, :1], requested="gradient", method="blake")
        for method in ("blake", "fmm"):
            empty = blake.fields(y, f, np.zeros((3, 0)), requested="traction",
                                 normals=np.zeros((3, 0)), method=method)
            self.assertEqual(empty["grad_u"].shape, (3, 3, 0))
            self.assertEqual(empty["traction"].shape, (3, 0))
            empty = blake.fields(np.zeros((3, 0)), np.zeros((3, 0)), x,
                                 requested="gradient", method=method)
            np.testing.assert_array_equal(empty["grad_u"], 0)
            self.assertEqual(empty["pressure"].shape, (3,))

    def test_all_direct_near_wall_fields_need_no_fmm(self):
        """Fully selected near-wall targets use only direct fields and no FMM3D import."""
        y = np.array([[0, .5], [0, .5], [1e-6, 1]])
        f = np.array([[1., 2], [3, -1], [2, 1]])
        x = np.array([[0.], [0], [0]])
        with patch.dict("sys.modules", {"fmm3dpy": None}):
            value, info = blake.fields(y, f, x, requested="gradient",
                                       correct_near_wall=True, return_info=True)
        reference = blake.fields(y, f, x, requested="gradient", method="blake")
        for name in reference:
            np.testing.assert_array_equal(value[name], reference[name])
        self.assertEqual(info["fmm_call_count"], 0)


@unittest.skipIf(fmm3dpy is None, "FMM3D unavailable: select the interpreter containing fmm3dpy")
class FMMFieldTests(unittest.TestCase):
    def test_fmm_self_test(self):
        """Run the FMM field self-test against direct Blake derivatives."""
        fmm_fields()

    def test_fmm_complete_fields_on_tree(self):
        """Verify FMM pressure, tensor ordering and traction against direct Blake on a tree."""
        rng = np.random.default_rng(151)
        y = np.vstack((rng.random((2, 1200)), .2 + rng.random(1200)))
        f = rng.standard_normal((3, 1200))
        x = np.vstack((rng.random((2, 19)), np.r_[np.zeros(5), .1 + rng.random(14)]))
        normals = np.tile([[0.], [0], [1]], (1, 19))
        options = dict(requested="traction", normals=normals, mu=1.7)
        direct = blake.fields(y, f, x, method="blake", **options)
        value, info = blake.fields(y, f, x, method="fmm", tol=1e-10, return_info=True, **options)
        for name in ("velocity", "pressure", "grad_u", "traction"):
            error = np.linalg.norm(value[name] - direct[name]) / np.linalg.norm(direct[name])
            self.assertLess(error, 1e-8, name)
        scale = np.linalg.norm(direct["grad_u"], axis=(0, 1))
        self.assertLess(np.max(np.abs(np.trace(value["grad_u"])) / scale), 1e-8)
        self.assertEqual((info["fmm_call_count"], info["stokes_target_flag"],
                          info["laplace_target_flag"]), (2, 3, 3))
        _, info = blake.fields(y, f, x, requested="pressure", return_info=True)
        self.assertEqual((info["stokes_target_flag"], info["laplace_target_flag"]), (2, 2))
        _, info = blake.velocity(y, f, x, return_info=True)
        self.assertEqual((info["stokes_target_flag"], info["laplace_target_flag"]), (1, 2))

    def test_mixed_near_wall_fields(self):
        """Selected targets receive exact direct fields; the rest come from FMM."""
        rng = np.random.default_rng(384)
        y = np.vstack((rng.random((2, 40)), 1e-6 * (1 + rng.random(40))))
        y[:, -1] = [.7, .6, .8]
        x = np.column_stack((y[:, :3], [.2, .3, 1.], [.9, .1, .5]))
        x[2, :3] = [0, 1e-10, 1e-7]
        f = rng.standard_normal(y.shape)
        value, info = blake.fields(y, f, x, requested="gradient", correct_near_wall=True,
                                   tol=1e-10, return_info=True)
        direct = blake.fields(y, f, x, requested="gradient", method="blake")
        selected = np.zeros(x.shape[1], bool)
        selected[:3] = True
        self.assertEqual(info["corrected_target_count"], 3)
        for name in direct:
            np.testing.assert_array_equal(value[name][..., selected], direct[name][..., selected])
            error = (np.linalg.norm(value[name][..., ~selected] - direct[name][..., ~selected])
                     / np.linalg.norm(direct[name][..., ~selected]))
            self.assertLess(error, 1e-8, name)


if __name__ == "__main__":
    unittest.main()
