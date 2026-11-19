"""Compare the FMM3D half-space wall velocity with STKFMM's ``StkWallFMM`` Stokeslet.

This is a cross-repository benchmark, not part of the ``blake`` package. It runs
in two interpreters because STKFMM's PySTKFMM/mpi4py environment does not
provide ``fmm3dpy``:

* the orchestrator runs in an interpreter with OpenMP-enabled ``fmm3dpy``,
  NumPy and Numba, times this repository's default two-call FMM path, and
  starts a worker;
* the worker (``--worker``) runs under STKFMM's interpreter with the paths set
  by ``STKFMM/Python/use-local.sh`` and times ``PySTKFMM.StkWallFMM``.

Both evaluators receive identical seeded inputs (hashed in each interpreter)
from RigidMultiblobsWall's ``faster_mobility/blakeComparison.py`` fixture and
are checked against this repository's independent direct Blake reference at the
same sampled targets. STKFMM's wall kernel requires the box's lower z-face to be
the wall and every point to lie in the lower half of its cubic box; its
``KERNEL.Stokes`` result contains 1/(8*pi) but not the viscosity.

STKFMM sources (https://github.com/wenyan4work/STKFMM, Apache-2.0) and
RigidMultiblobsWall (https://github.com/annabroms/RigidMultiblobsWall, GPL-3.0)
are local checkouts selected with ``--stkfmm-root`` and ``--rigid-root``.
"""

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import platform
from statistics import median
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
RIGID_DRIVER = Path("faster_mobility/blakeComparison.py")
STKFMM_PYTHON = Path("build/conda/bin/python")
STKFMM_LIB = Path("build/local/lib")
SEED_BASE = 1927
# Small validation fixtures: (name, ns, nt, source height, wall height, viscosity).
VALIDATION = [("wall", 23, 21, .2, 0., 1.), ("tree", 2048, 61, .2, 0., 1.),
              ("shifted_wall", 23, 29, .01, 1.3, 1.7)]


def arguments():
    """Parse settings shared by the orchestrator and the STKFMM worker."""
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--stkfmm-root", type=Path, default=ROOT.parent / "STKFMM")
    parser.add_argument("--rigid-root", type=Path, default=ROOT.parent / "RigidMultiblobsWall")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 32768, 131072])
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-6, 1e-8],
                        help="FMM3D requested tolerances")
    parser.add_argument("--orders", type=int, nargs="+", default=[6, 8, 10, 12],
                        help="STKFMM/PVFMM multipole orders (even)")
    parser.add_argument("--max-pts", type=int, default=500,
                        help="STKFMM maximum points per leaf box")
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--reference-targets", type=int, default=256)
    parser.add_argument("--output", type=Path, default=ROOT / "results/stkfmm_wall_comparison")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if (args.threads < 1 or args.repeats < 3 or args.max_pts < 1 or args.reference_targets < 1
            or any(n < 1 for n in args.sizes)):
        parser.error("Use positive counts/threads/max_pts and at least three repetitions")
    if any(not 0 < tol < 1 for tol in args.tolerances):
        parser.error("Tolerances must lie strictly between zero and one")
    if any(order < 2 or order % 2 for order in args.orders):
        parser.error("STKFMM orders must be even and at least two")
    args.sizes, args.orders = sorted(set(args.sizes)), sorted(set(args.orders))
    for name in ("stkfmm_root", "rigid_root", "output"):
        setattr(args, name, getattr(args, name).resolve())
    return args


def load_module(path, name):
    """Import an external driver by path without adding its directory to sys.path."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(samples):
    """Retain all samples with median and observed range."""
    return dict(median_seconds=median(samples), min_seconds=min(samples),
                max_seconds=max(samples), samples_seconds=samples)


def timed(evaluate, repeats, label):
    """Warm the exact case once, then time complete repeated evaluations."""
    evaluate()
    samples, value, info = [], None, []
    for repetition in range(repeats):
        start = perf_counter()
        value, extra = evaluate()
        samples.append(perf_counter() - start)
        info.append(extra)
        print(f"{label}, sample {repetition + 1}/{repeats}: {samples[-1]:.6f}s", flush=True)
    return value, summary(samples), info


def digest(np, *arrays):
    """Hash float64 inputs so both interpreters can prove they used the same geometry."""
    hasher = hashlib.sha256()
    for array in arrays:
        hasher.update(np.ascontiguousarray(array, dtype=np.float64).tobytes())
    return hasher.hexdigest()


def write_json(path, value):
    """Write a report atomically so an interrupted run never leaves partial JSON."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def sample_indices(np, n, count):
    """Choose evenly spaced reference targets shared by both evaluators."""
    return np.unique(np.linspace(0, n - 1, min(count, n), dtype=int))


def openmp_threads(library):
    """Query omp_get_max_threads from an already loaded OpenMP runtime."""
    import ctypes
    function = ctypes.CDLL(library).omp_get_max_threads
    function.restype, function.argtypes = ctypes.c_int, []
    return function()


class WallSTKFMM:
    """Physical wall Stokeslet velocity from PySTKFMM.StkWallFMM for 3-by-N inputs.

    Construction loads PVFMM's precomputed operators and is timed separately.
    Each call sets a box whose lower z-face is the wall, inserts points and
    rebuilds every tree, as the FMM3D path does; ``reevaluate`` reuses them.
    """

    def __init__(self, np, module, order, max_pts):
        self.np, self.module, self.kernel = np, module, module.KERNEL.Stokes
        self.empty = np.zeros((0, 3))
        start = perf_counter()
        self.fmm = module.StkWallFMM(order, max_pts, module.PAXIS.NONE, self.kernel)
        self.construct_seconds = perf_counter() - start

    def box(self, sources, targets, wall):
        """Return the half-cube box origin/length required by StkWallFMM."""
        np = self.np
        if sources.shape[1] and sources[2].min() <= wall:
            raise ValueError("Sources must lie strictly above the wall")
        if targets.shape[1] and targets[2].min() < wall:
            raise ValueError("Targets must not lie below the wall")
        points = np.hstack((sources, targets))
        lower, upper = points[:2].min(axis=1), points[:2].max(axis=1)
        length = max(float((upper - lower).max()), 2 * float(points[2].max() - wall)) or 1.0
        pad = 1e-3 * length  # Points must lie strictly inside [0,1)^2 x [0,0.5).
        return np.array([lower[0] - pad, lower[1] - pad, wall]), length + 2 * pad

    def __call__(self, sources, forces, targets, mu=1.0, wall=0.0):
        """Evaluate with a fresh box, point set and trees; return 3-by-Nt velocity."""
        np = self.np
        origin, length = self.box(sources, targets, wall)
        self.fmm.set_box(origin, length)
        self.fmm.set_points(np.ascontiguousarray(sources.T), np.ascontiguousarray(targets.T),
                            self.empty)
        self.fmm.setup_tree(self.kernel)
        self.nt = targets.shape[1]
        return self.reevaluate(forces, mu)

    def reevaluate(self, forces, mu=1.0):
        """Evaluate new forces on the trees from the previous call."""
        np = self.np
        output = np.zeros((self.nt, 3))
        self.fmm.clear_fmm(self.kernel)
        self.fmm.evaluate_fmm(self.kernel, np.ascontiguousarray(forces.T), output, self.empty)
        return output.T / mu


def worker(args):
    """Validate and time StkWallFMM under STKFMM's interpreter; write worker.json."""
    import numpy as np
    sys.path.insert(0, str(ROOT / "python"))
    import blake
    import PySTKFMM

    comparison = load_module(args.rigid_root / RIGID_DRIVER, "rigid_blake_comparison")
    threads = openmp_threads("libgomp.so.1")
    if threads != args.threads:
        raise SystemExit(f"STKFMM OpenMP reports {threads} threads, expected {args.threads}")
    validation, cases = [], []
    for order in args.orders:
        fmm = WallSTKFMM(np, PySTKFMM, order, args.max_pts)
        for name, ns, nt, height, wall, mu in VALIDATION:
            y, f, x = comparison.fixture(np, ns, nt, 9271 + ns, height=height, wall=wall)
            x[2, :3] = wall  # Include targets on the wall; their residuals are reported raw.
            geom = blake.prepare(y, x, wall_height=wall)
            reference = blake.direct_velocity(geom, f, mu=mu)
            value = fmm(y, f, x, mu=mu, wall=wall)
            error = blake.metrics(value, reference, blake.constituent_scale(geom, f, mu=mu))
            comparison.check_metrics(error, normalized=1e-3)
            validation.append(dict(order=order, fixture=name, error=error,
                                   wall_residual_max=float(np.abs(value[:, :3]).max())))
            print(f"order {order} {name}: {error}", flush=True)
    for case, n in enumerate(args.sizes):
        y, f, x = comparison.fixture(np, n, n, SEED_BASE + case)
        indices = sample_indices(np, n, args.reference_targets)
        geom = blake.prepare(y, x[:, indices])
        reference = blake.direct_velocity(geom, f)
        scale = blake.constituent_scale(geom, f)
        for order in args.orders:
            fmm = WallSTKFMM(np, PySTKFMM, order, args.max_pts)
            label = f"N={n} STKFMM order={order}"
            value, full, _ = timed(lambda: (fmm(y, f, x), {}), args.repeats, label)
            reused, reuse, _ = timed(lambda: (fmm.reevaluate(f), {}), args.repeats,
                                     label + " reused tree")
            if not np.array_equal(value, reused):
                print(f"{label}: reused-tree result differs by "
                      f"{np.abs(value - reused).max():.3e}", flush=True)
            error = blake.metrics(value[:, indices], reference, scale)
            comparison.check_metrics(error, normalized=1e-3)
            cases.append(dict(ns=n, nt=n, order=order, max_pts=args.max_pts,
                              input_sha256=digest(np, y, f, x),
                              construct_seconds=fmm.construct_seconds,
                              full=full, reused_tree=reuse, error=error))
            print(f"{label}: {full['median_seconds']:.4f}s, reuse "
                  f"{reuse['median_seconds']:.4f}s, error {error['global_relative']:.3e}",
                  flush=True)
    write_json(args.output / "worker.json", dict(
        validation=validation, cases=cases, openmp_threads=threads,
        provenance=dict(python=sys.version, executable=sys.executable,
                        numpy=np.__version__, pystkfmm=str(Path(PySTKFMM.__file__).resolve()),
                        library=str((args.stkfmm_root / STKFMM_LIB / "libSTKFMM_SHARED.so").resolve()),
                        stkfmm_commit=git_head(args.stkfmm_root))))


def git_head(path):
    """Record a checkout's branch and commit, or None outside git."""
    try:
        run = lambda *a: subprocess.run(["git", "-C", str(path), *a], capture_output=True,
                                        text=True, check=True).stdout.strip()
        return dict(branch=run("rev-parse", "--abbrev-ref", "HEAD"), commit=run("rev-parse", "HEAD"))
    except (OSError, subprocess.CalledProcessError):
        return None


def stkfmm_environment(args):
    """Reproduce STKFMM/Python/use-local.sh for the worker with fixed thread controls."""
    root = args.stkfmm_root
    if not (root / STKFMM_PYTHON).is_file() or not (root / STKFMM_LIB / "python/PySTKFMM.py").is_file():
        raise SystemExit(f"No local STKFMM build under {root}; see STKFMM/Python/README.md")
    env = dict(os.environ)
    env.pop("PYTHONHOME", None)
    env.update(PYTHONPATH=str(root / STKFMM_LIB / "python"),
               LD_LIBRARY_PATH=f"{root / STKFMM_LIB}:{root / 'build/conda/lib'}",
               PVFMM_DIR=str(root / "build/local"),
               PATH=f"{root / 'build/conda/bin'}:{env.get('PATH', '')}")
    return env


def main():
    """Time FMM3D, run the STKFMM worker on identical inputs and merge both reports."""
    args = arguments()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), NUMBA_NUM_THREADS=str(args.threads),
                      OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1")
    if args.worker:
        return worker(args)
    if not (args.rigid_root / RIGID_DRIVER).is_file():
        raise SystemExit(f"Missing {args.rigid_root / RIGID_DRIVER}; pass --rigid-root")
    env = stkfmm_environment(args)
    sys.path.insert(0, str(ROOT / "python"))
    import numpy as np
    import fmm3dpy
    import blake

    comparison = load_module(args.rigid_root / RIGID_DRIVER, "rigid_blake_comparison")
    openmp = comparison.inspect_fmm_openmp(Path(fmm3dpy.__file__).parent)
    assert all(item.get("openmp_max_threads") == args.threads for item in openmp["extensions"])
    cpu = next((line.split(":", 1)[1].strip() for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")), platform.processor())
    report = dict(
        status="inProgress", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        settings={k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()
                  if k != "worker"},
        geometry="unit-square lateral clouds; source heights [0.2,0.4]; target heights "
                 "[0.1,1.1]; wall=0; mu=1; Ns=Nt; RigidMultiblobsWall fixture seeds 1927+i",
        timing_scope="one complete evaluation per sample after a warmup. FMM3D: blake.apply on "
                     "prepared geometry (guard, packing, two native calls, assembly). STKFMM: box, "
                     "point-major copies, set_points, setup_tree, clear, evaluate, viscosity "
                     "scaling; operator construction excluded. reused_tree: clear+evaluate only.",
        provenance=dict(cpu=cpu, python=sys.version, executable=sys.executable,
                        packages={name: importlib.metadata.version(name)
                                  for name in ("numpy", "numba", "fmm3dpy")},
                        halfspace_commit=git_head(ROOT), rigid_multiblobs=git_head(args.rigid_root),
                        fmm_openmp=openmp),
        fmm3d_cases=[], stkfmm=None)
    for case, n in enumerate(args.sizes):
        y, f, x = comparison.fixture(np, n, n, SEED_BASE + case)
        indices = sample_indices(np, n, args.reference_targets)
        geom = blake.prepare(y, x[:, indices])
        reference = blake.direct_velocity(geom, f)
        scale = blake.constituent_scale(geom, f)
        prepared = blake.prepare(y, x)
        for tolerance in args.tolerances:
            value, timing, phases = timed(
                lambda: blake.apply(prepared, f, method="fmm", tol=tolerance,
                                    correct_near_wall=False, diagnostics=False, return_info=True),
                args.repeats, f"N={n} FMM3D tol={tolerance:g}")
            error = blake.metrics(value[:, indices], reference, scale)
            comparison.check_metrics(error, normalized=100 * tolerance + 1e-12)
            report["fmm3d_cases"].append(dict(
                ns=n, nt=n, tolerance=tolerance, input_sha256=digest(np, y, f, x),
                full=timing, error=error,
                phase_medians={key: median(p[key] for p in phases) for key in
                               ("guard_seconds", "stokes_seconds", "laplace_seconds",
                                "assembly_seconds")}))
            write_json(args.output / "comparison.json", report)
    del prepared, value
    print("Starting STKFMM worker", flush=True)
    subprocess.run([str(args.stkfmm_root / STKFMM_PYTHON), str(Path(__file__).resolve()),
                    "--worker", *sys.argv[1:]], env=env, check=True)
    stk = json.loads((args.output / "worker.json").read_text())
    hashes = {c["ns"]: c["input_sha256"] for c in report["fmm3d_cases"]}
    if any(hashes[c["ns"]] != c["input_sha256"] for c in stk["cases"]):
        raise AssertionError("STKFMM worker inputs differ from the FMM3D inputs")
    report.update(stkfmm=stk, status="passed")
    write_json(args.output / "comparison.json", report)
    (args.output / "worker.json").unlink()
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        out = csv.writer(stream)
        out.writerow(["code", "ns", "setting", "median_seconds", "reused_tree_median_seconds",
                      "global_relative_error", "constituent_normalized_error"])
        for c in report["fmm3d_cases"]:
            out.writerow(["FMM3D", c["ns"], f"tol={c['tolerance']:g}", c["full"]["median_seconds"],
                          "", c["error"]["global_relative"], c["error"]["constituent_normalized"]])
        for c in stk["cases"]:
            out.writerow(["STKFMM", c["ns"], f"order={c['order']}", c["full"]["median_seconds"],
                          c["reused_tree"]["median_seconds"], c["error"]["global_relative"],
                          c["error"]["constituent_normalized"]])
    print((args.output / "comparison.csv").read_text())


if __name__ == "__main__":
    main()
