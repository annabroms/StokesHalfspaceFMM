"""Reproducible numerical validation; this module does not run benchmark profiles."""

import importlib
import importlib.metadata
import os
from pathlib import Path
import platform
import sys
from time import perf_counter

import numpy as np

from .diagnostics import constituent_scale, metrics
from .evaluate import velocity
from .fmm import _checked_output
from .geometry import prepare


def validate_wrappers():
    """Compare real FMM calls with independent free-space Stokes and Laplace kernels."""
    import fmm3dpy

    rng = np.random.default_rng(93)
    sources = rng.random((3, 1200))
    targets = 1.2 + rng.random((3, 31))
    forces = rng.standard_normal((3, 1200))
    charges = rng.standard_normal(1200)
    dipoles = rng.standard_normal((3, 1200))
    target_count = targets.shape[1]

    # Evaluate one target at a time, without a full interaction array.
    reference = np.zeros_like(targets)
    for target_index in range(target_count):
        r = targets[:, target_index, None] - sources
        inverse_r = 1 / np.linalg.norm(r, axis=0)
        r_dot_force = np.sum(r * forces, axis=0)
        stokeslets = (forces * inverse_r + r * r_dot_force * inverse_r**3) / (8 * np.pi)
        reference[:, target_index] = np.sum(stokeslets, axis=1)

    stokes = fmm3dpy.stfmm3d(
        eps=1e-12,
        sources=sources,
        stoklet=forces,
        targets=targets,
        ifppreg=0,
        ifppregtarg=1,
    )
    value = _checked_output(stokes, "pottarg", targets.shape, "stfmm3d")
    report = {
        "stokes_relative": float(
            np.linalg.norm(value - reference) / np.linalg.norm(reference)
        ),
    }

    harmonic_cases = (
        ("charge", charges, np.zeros_like(dipoles)),
        ("dipole", np.zeros_like(charges), dipoles),
        ("combined", charges, dipoles),
    )
    for name, case_charges, case_dipoles in harmonic_cases:
        potential = np.zeros(target_count)
        gradient = np.zeros_like(targets)
        for target_index in range(target_count):
            r = targets[:, target_index, None] - sources
            inverse_r = 1 / np.linalg.norm(r, axis=0)
            r_dot_dipole = np.sum(r * case_dipoles, axis=0)
            potential[target_index] = np.sum(
                (case_charges * inverse_r + r_dot_dipole * inverse_r**3) / (4 * np.pi)
            )
            gradient[:, target_index] = np.sum(
                (
                    -case_charges * r * inverse_r**3
                    + case_dipoles * inverse_r**3
                    - 3 * r * r_dot_dipole * inverse_r**5
                ) / (4 * np.pi),
                axis=1,
            )

        harmonic = fmm3dpy.lfmm3d(
            eps=1e-12,
            sources=sources,
            charges=case_charges,
            dipvec=case_dipoles,
            targets=targets,
            pg=0,
            pgt=2,
        )
        fmm_potential = _checked_output(
            harmonic, "pottarg", (target_count,), "lfmm3d"
        )
        fmm_gradient = _checked_output(
            harmonic, "gradtarg", targets.shape, "lfmm3d"
        )
        report[name + "_potential_relative"] = float(
            np.linalg.norm(fmm_potential - potential) / np.linalg.norm(potential)
        )
        report[name + "_gradient_relative"] = float(
            np.linalg.norm(fmm_gradient - gradient) / np.linalg.norm(gradient)
        )

    if max(report.values()) > 1e-10:
        raise AssertionError(f"FMM3D normalization probe failed: {report}")
    return report


def provenance():
    """Record imported wrapper paths and software versions."""
    import fmm3dpy
    from . import fmm

    paths = {"wrapper": Path(fmm3dpy.__file__).parent / "fmm3d.py"}
    for module_name in ("stfmm3d_fortran", "lfmm3d_fortran"):
        module = importlib.import_module("fmm3dpy." + module_name)
        paths[module_name] = Path(module.__file__)

    return {
        "python": sys.version,
        "executable": sys.executable,
        "numpy": np.__version__,
        "fmm3dpy": importlib.metadata.version("fmm3dpy"),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "omp_num_threads": os.getenv("OMP_NUM_THREADS", "unset"),
        "files": {
            name: {"path": str(path.resolve())} for name, path in paths.items()
        },
        "adapter_path": str(Path(fmm.__file__).resolve()),
        "source_binary_relationship": (
            "No build-source relationship inferred from a checkout revision"
        ),
    }


def validate_field_cases(rng=None):
    """Compare FMM and direct fields in four geometries, including raw wall residuals."""
    if rng is None:
        rng = np.random.default_rng(927)

    # Rows specify name, Ns, Nt, source height, wall, viscosity, horizontal shift.
    geometries = (
        ("bulk and wall", 1200, 81, 0.2, 0.0, 1.0, 0.0),
        ("shallow sources", 73, 41, 1e-6, 0.0, 1.0, 0.0),
        ("shifted wall and viscosity", 73, 41, 0.01, 1.3, 1.7, 0.0),
        ("horizontal translation", 73, 41, 0.2, 0.0, 1.0, 1e6),
    )
    cases = []
    for name, source_count, target_count, height, wall, mu, shift in geometries:
        sources = np.vstack((
            rng.random((2, source_count)) + shift,
            wall + height * (1 + rng.random(source_count)),
        ))
        targets = np.vstack((
            rng.random((2, target_count)) + shift,
            wall + 0.1 + rng.random(target_count),
        ))
        wall_count = min(13, target_count // 3)
        wall_targets = slice(0, wall_count)
        near_wall_targets = slice(wall_count, 2 * wall_count)
        bulk_targets = slice(2 * wall_count, None)
        targets[2, wall_targets] = wall
        targets[2, near_wall_targets] = wall + 1e-10

        # Include wall points directly beneath shallow sources.
        targets[:2, wall_targets] = sources[:2, wall_targets]
        forces = rng.standard_normal((3, source_count))
        reference, direct_info = velocity(
            sources, forces, targets,
            method="blake", wall_height=wall, mu=mu,
            diagnostics=True, return_info=True,
        )
        scale = direct_info["constituent_scale"]
        zero_wall_reference = np.zeros((3, wall_count))

        for tol in (1e-6, 1e-9, 1e-12):
            value, info = velocity(
                sources, forces, targets,
                wall_height=wall, mu=mu, tol=tol, return_info=True,
            )
            errors = metrics(value, reference, scale)
            bulk_errors = metrics(
                value[:, bulk_targets], reference[:, bulk_targets], scale[bulk_targets]
            )
            wall_errors = metrics(
                value[:, wall_targets], zero_wall_reference, scale[wall_targets]
            )
            direct_wall_errors = metrics(
                reference[:, wall_targets], zero_wall_reference, scale[wall_targets]
            )

            normalized_budget = 100 * tol + 1e-12
            bulk_relative_budget = 100 * tol + 1e-11
            if (
                errors["constituent_normalized"] > normalized_budget
                or wall_errors["constituent_normalized"] > normalized_budget
            ):
                raise AssertionError(
                    f"Field or wall acceptance failed for {name}, tol={tol}"
                )
            if (
                direct_wall_errors["constituent_normalized"] > 1e-12
                or bulk_errors["global_relative"] > bulk_relative_budget
            ):
                raise AssertionError(
                    f"Direct wall or bulk relative acceptance failed for {name}"
                )

            direct_seconds = (
                direct_info["total_seconds"] - direct_info["diagnostics_seconds"]
            )
            cases.append({
                "geometry": name,
                "ns": source_count,
                "nt": target_count,
                "wall_targets": wall_count,
                "near_wall_targets": wall_count,
                "source_height_base": height,
                "wall_height": wall,
                "horizontal_shift": shift,
                "mu": mu,
                "tol": tol,
                "field": errors,
                "bulk": bulk_errors,
                "wall": wall_errors,
                "wall_component_max": np.max(
                    np.abs(value[:, wall_targets]), axis=1
                ).tolist(),
                "direct_wall": direct_wall_errors,
                "direct_seconds": direct_seconds,
                "diagnostics_seconds": direct_info["diagnostics_seconds"],
                "fmm_seconds": info["total_seconds"],
                "stokes_seconds": info["stokes_seconds"],
                "laplace_seconds": info["laplace_seconds"],
                "fmm_call_count": info["fmm_call_count"],
            })
    return cases


def validate_reciprocity(rng=None):
    """Check pair tensors and distributed force work after swapping two point clouds."""
    if rng is None:
        rng = np.random.default_rng(927)

    points_a = np.vstack((rng.random((2, 1200)), 0.2 + rng.random(1200)))
    points_b = np.vstack((rng.random((2, 701)), 0.3 + rng.random(701)))
    forces_a = rng.standard_normal(points_a.shape)
    forces_b = rng.standard_normal(points_b.shape)

    direct_timer = perf_counter()
    direct_work_ab = forces_b * velocity(points_a, forces_a, points_b, method="blake")
    direct_work_ba = forces_a * velocity(points_b, forces_b, points_a, method="blake")
    direct_defect = float(
        abs(np.sum(direct_work_ab) - np.sum(direct_work_ba))
        / (np.sum(np.abs(direct_work_ab)) + np.sum(np.abs(direct_work_ba)))
    )
    direct_reciprocity = {
        "ns_a": 1200,
        "ns_b": 701,
        "distributed_defect": direct_defect,
        "seconds": perf_counter() - direct_timer,
    }
    if direct_defect > 1e-12:
        raise AssertionError("Independent direct distributed reciprocity failed")

    reciprocity = []
    for tol in (1e-6, 1e-9, 1e-12):
        velocity_at_b = velocity(points_a, forces_a, points_b, tol=tol)
        velocity_at_a = velocity(points_b, forces_b, points_a, tol=tol)
        work_ab = forces_b * velocity_at_b
        work_ba = forces_a * velocity_at_a
        denominator = np.sum(np.abs(work_ab)) + np.sum(np.abs(work_ba))
        defect = float(abs(np.sum(work_ab) - np.sum(work_ba)) / denominator)

        # Unit forces supply the three tensor columns for each direction.
        source, target = points_a[:, :1], points_b[:, :1]
        cartesian_forces = np.eye(3)
        forward = np.zeros((3, 3))
        reverse = np.zeros((3, 3))
        for direction in range(3):
            unit_force = cartesian_forces[:, direction : direction + 1]
            forward[:, direction] = velocity(source, unit_force, target, tol=tol)[:, 0]
        for direction in range(3):
            unit_force = cartesian_forces[:, direction : direction + 1]
            reverse[:, direction] = velocity(target, unit_force, source, tol=tol)[:, 0]
        pair_defect = float(
            np.linalg.norm(forward - reverse.T)
            / (np.linalg.norm(forward) + np.linalg.norm(reverse))
        )

        # Multi-source Cartesian loads expose direction-specific sign errors.
        for direction in range(3):
            forces = np.zeros_like(forces_a)
            forces[direction] = forces_a[direction]
            reference = velocity(points_a, forces, points_b[:, :19], method="blake")
            value = velocity(points_a, forces, points_b[:, :19], tol=tol)
            scale = constituent_scale(prepare(points_a, points_b[:, :19]), forces)
            errors = metrics(value, reference, scale)
            if errors["constituent_normalized"] > 100 * tol + 1e-12:
                raise AssertionError("Cartesian force acceptance failed")

        if max(defect, pair_defect) > 100 * tol + 1e-12:
            raise AssertionError(f"Reciprocity failed at tol={tol}")
        reciprocity.append({
            "tol": tol,
            "ns_a": 1200,
            "ns_b": 701,
            "distributed_defect": defect,
            "pair_defect": pair_defect,
        })
    return reciprocity, direct_reciprocity


def validate_velocity():
    """Run field, wall and reciprocity checks with one reproducible random stream."""
    rng = np.random.default_rng(927)
    start = perf_counter()
    cases = validate_field_cases(rng)
    reciprocity, direct_reciprocity = validate_reciprocity(rng)
    return {
        "seed": 927,
        "cases": cases,
        "reciprocity": reciprocity,
        "direct_reciprocity": direct_reciprocity,
        "measured_seconds": perf_counter() - start,
        "timing_note": "Single correctness evaluations, no warmup or speedup claim",
    }
