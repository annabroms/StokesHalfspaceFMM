"""Reproducible accuracy and runtime measurements; no benchmark profiles are launched."""

import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .diagnostics import constituent_scale, metrics
from .evaluate import velocity
from .fmm import _output
from .geometry import prepare


def validate_wrappers():
    """Compare real FMM calls with independent free-space kernels beyond tiny leaf sums."""
    import fmm3dpy
    rng = np.random.default_rng(93)
    y, x = rng.random((3, 1200)), 1.2 + rng.random((3, 31))
    f, q, d = rng.standard_normal((3, 1200)), rng.standard_normal(1200), rng.standard_normal((3, 1200))
    ref = np.zeros_like(x)
    # Use one target workspace, never a full source-by-target interaction array.
    for m in range(x.shape[1]):
        r = x[:, m, None] - y
        a = 1 / np.linalg.norm(r, axis=0)
        ref[:, m] = np.sum((f*a + r*np.sum(r*f, axis=0)*a**3) / (8*np.pi), axis=1)
    out = fmm3dpy.stfmm3d(eps=1e-12, sources=y, stoklet=f, targets=x, ifppreg=0, ifppregtarg=1)
    report = {"stokes_relative": float(np.linalg.norm(_output(out, "pottarg", x.shape, "stfmm3d") - ref) / np.linalg.norm(ref))}
    for name, charges, dipvec in (("charge", q, np.zeros_like(d)), ("dipole", np.zeros_like(q), d), ("combined", q, d)):
        phi, grad = np.zeros(x.shape[1]), np.zeros_like(x)
        for m in range(x.shape[1]):
            r = x[:, m, None] - y
            a = 1 / np.linalg.norm(r, axis=0)
            rd = np.sum(r*dipvec, axis=0)
            phi[m] = np.sum((charges*a + rd*a**3) / (4*np.pi))
            grad[:, m] = np.sum((-charges*r*a**3 + dipvec*a**3 - 3*r*rd*a**5) / (4*np.pi), axis=1)
        out = fmm3dpy.lfmm3d(eps=1e-12, sources=y, charges=charges, dipvec=dipvec, targets=x, pg=0, pgt=2)
        report[name + "_potential_relative"] = float(np.linalg.norm(_output(out, "pottarg", (31,), "lfmm3d") - phi) / np.linalg.norm(phi))
        report[name + "_gradient_relative"] = float(np.linalg.norm(_output(out, "gradtarg", x.shape, "lfmm3d") - grad) / np.linalg.norm(grad))
    if max(report.values()) > 1e-10:
        raise AssertionError(f"FMM3D normalization probe failed: {report}")
    return report


def provenance():
    """Record imported wrapper paths and software versions without modifying dependencies."""
    import fmm3dpy
    from . import fmm
    paths = {"wrapper": Path(fmm3dpy.__file__).parent / "fmm3d.py"}
    for name in ("stfmm3d_fortran", "lfmm3d_fortran"):
        module = __import__("fmm3dpy." + name, fromlist=[name])
        paths[name] = Path(module.__file__)
    files = {name: dict(path=str(path.resolve())) for name, path in paths.items()}
    return dict(python=sys.version, executable=sys.executable, numpy=np.__version__,
                fmm3dpy=importlib.metadata.version("fmm3dpy"), platform=platform.platform(),
                machine=platform.machine(), omp_num_threads=os.getenv("OMP_NUM_THREADS", "unset"),
                files=files, adapter_path=str(Path(fmm.__file__).resolve()),
                source_binary_relationship="No build-source relationship inferred from a checkout revision")


def validate_velocity():
    """Check fields, raw wall residuals and two reciprocity identities over a tolerance sweep."""
    rng = np.random.default_rng(927)
    cases = []
    start = perf_counter()
    # One representative fixture has enough sources to exercise multipole evaluation.
    for name, ns, nt, height, wall, mu, horizontal in (
        ("bulk and wall", 1200, 81, .2, 0., 1., 0.),
        ("shallow sources", 73, 41, 1e-6, 0., 1., 0.),
        ("shifted wall and viscosity", 73, 41, .01, 1.3, 1.7, 0.),
        ("horizontal translation", 73, 41, .2, 0., 1., 1e6),
    ):
        y = np.vstack((rng.random((2, ns)) + horizontal, wall + height*(1 + rng.random(ns))))
        x = np.vstack((rng.random((2, nt)) + horizontal, wall + .1 + rng.random(nt)))
        nw = min(13, nt // 3)
        x[2, :nw] = wall
        x[2, nw:2*nw] = wall + 1e-10
        # Include wall points directly beneath shallow sources.
        x[:2, :nw] = y[:2, :nw]
        f = rng.standard_normal((3, ns))
        ref, direct_info = velocity(y, f, x, method="blake", wall_height=wall, mu=mu,
                                    diagnostics=True, return_info=True)
        scale = direct_info["constituent_scale"]
        for tol in (1e-6, 1e-9, 1e-12):
            u, info = velocity(y, f, x, wall_height=wall, mu=mu, tol=tol, return_info=True)
            errors = metrics(u, ref, scale)
            bulk = metrics(u[:, 2*nw:], ref[:, 2*nw:], scale[2*nw:])
            wall_errors = metrics(u[:, :nw], np.zeros((3, nw)), scale[:nw])
            direct_wall = metrics(ref[:, :nw], np.zeros((3, nw)), scale[:nw])
            budget = 100*tol + 1e-12
            if errors["constituent_normalized"] > budget or wall_errors["constituent_normalized"] > budget:
                raise AssertionError(f"Field or wall acceptance failed for {name}, tol={tol}")
            if direct_wall["constituent_normalized"] > 1e-12 or bulk["global_relative"] > 100*tol + 1e-11:
                raise AssertionError(f"Direct wall or bulk relative acceptance failed for {name}")
            cases.append(dict(geometry=name, ns=ns, nt=nt, wall_targets=nw, near_wall_targets=nw,
                              source_height_base=height, wall_height=wall, horizontal_shift=horizontal,
                              mu=mu, tol=tol, field=errors, bulk=bulk, wall=wall_errors,
                              wall_component_max=np.max(np.abs(u[:, :nw]), axis=1).tolist(),
                              direct_wall=direct_wall,
                              direct_seconds=direct_info["total_seconds"] - direct_info["diagnostics_seconds"],
                              diagnostics_seconds=direct_info["diagnostics_seconds"],
                              fmm_seconds=info["total_seconds"], stokes_seconds=info["stokes_seconds"],
                              laplace_seconds=info["laplace_seconds"], fmm_call_count=info["fmm_call_count"]))
    reciprocity = []
    a = np.vstack((rng.random((2, 1200)), .2 + rng.random(1200)))
    b = np.vstack((rng.random((2, 701)), .3 + rng.random(701)))
    fa, fb = rng.standard_normal(a.shape), rng.standard_normal(b.shape)
    direct_timer = perf_counter()
    direct_ab = fb * velocity(a, fa, b, method="blake")
    direct_ba = fa * velocity(b, fb, a, method="blake")
    direct_defect = float(abs(np.sum(direct_ab) - np.sum(direct_ba)) /
                          (np.sum(np.abs(direct_ab)) + np.sum(np.abs(direct_ba))))
    direct_reciprocity = dict(ns_a=1200, ns_b=701, distributed_defect=direct_defect,
                             seconds=perf_counter() - direct_timer)
    if direct_defect > 1e-12:
        raise AssertionError("Independent direct distributed reciprocity failed")
    for tol in (1e-6, 1e-9, 1e-12):
        ua = velocity(a, fa, b, tol=tol)
        ub = velocity(b, fb, a, tol=tol)
        work_ab, work_ba = fb*ua, fa*ub
        denominator = np.sum(np.abs(work_ab)) + np.sum(np.abs(work_ba))
        defect = float(abs(np.sum(work_ab) - np.sum(work_ba)) / denominator)
        y, x = a[:, :1], b[:, :1]
        forward = np.column_stack([velocity(y, np.eye(3)[:, j:j+1], x, tol=tol)[:, 0] for j in range(3)])
        reverse = np.column_stack([velocity(x, np.eye(3)[:, j:j+1], y, tol=tol)[:, 0] for j in range(3)])
        pair_defect = float(np.linalg.norm(forward - reverse.T) / (np.linalg.norm(forward) + np.linalg.norm(reverse)))
        # Also use a multi-source Cartesian load to expose directional sign errors.
        for direction in range(3):
            f = np.zeros_like(fa)
            f[direction] = fa[direction]
            ref = velocity(a, f, b[:, :19], method="blake")
            value = velocity(a, f, b[:, :19], tol=tol)
            scale = constituent_scale(prepare(a, b[:, :19]), f)
            if metrics(value, ref, scale)["constituent_normalized"] > 100*tol + 1e-12:
                raise AssertionError("Cartesian force acceptance failed")
        if max(defect, pair_defect) > 100*tol + 1e-12:
            raise AssertionError(f"Reciprocity failed at tol={tol}")
        reciprocity.append(dict(tol=tol, ns_a=1200, ns_b=701, distributed_defect=defect, pair_defect=pair_defect))
    return dict(seed=927, cases=cases, reciprocity=reciprocity, direct_reciprocity=direct_reciprocity,
                measured_seconds=perf_counter() - start,
                timing_note="Single correctness evaluations, no warmup or speedup claim")
