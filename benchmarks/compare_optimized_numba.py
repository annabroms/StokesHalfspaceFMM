"""Rerun the historical eight-thread FMM comparison against specialized direct Numba.

This is a cross-repository benchmark, not part of the ``blake`` package. The
direct comparator and the shared fixture/validation helpers come from a local
checkout of the ``faster_mobility`` branch of RigidMultiblobsWall
(https://github.com/annabroms/RigidMultiblobsWall, GPL-3.0), selected with
``--rigid-root`` (default: a sibling directory ``../RigidMultiblobsWall``)::

    git clone --branch faster_mobility https://github.com/annabroms/RigidMultiblobsWall.git

That checkout must provide:

* ``mobility/blake_numba.py``: the specialized point-force Blake kernel
  ``_point_force_kernel`` used as the direct Numba comparator;
* ``faster_mobility/blakeComparison.py``: the original comparison driver, reused
  unchanged for its fixture, independent validation, error metrics, OpenMP
  inspection and JSON writer.

The interpreter must also provide NumPy, Numba, llvmlite and OpenMP-enabled
``fmm3dpy``; this repository's ``python/blake`` package is added to the path.
Neither the package nor its tests import RigidMultiblobsWall. Setup on a new
machine is described in docs/benchmarks.md#running-the-python-benchmarks-on-another-machine.
"""

import argparse
import csv
from datetime import datetime, timezone
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
from statistics import median
import sys
from time import perf_counter
from types import SimpleNamespace

import rigid_multiblobs as rigid

RIGID_URL, RIGID_KERNEL, RIGID_DRIVER = rigid.URL, rigid.KERNEL, rigid.DRIVER


def load_comparison(path):
    """Reuse the original fixture and independent validation without altering its driver."""
    spec = importlib.util.spec_from_file_location("historical_blake_comparison", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def summary(samples):
    """Retain all paired samples with median and observed range."""
    return dict(median_seconds=median(samples), min_seconds=min(samples),
                max_seconds=max(samples), samples_seconds=samples)


def timed_pairs(setup, evaluate, repeats, label):
    """Warm the exact case, then time setup and application within each complete solve."""
    print(f"{label}: warming full case", flush=True)
    evaluate(setup())
    print(f"{label}: warmup complete", flush=True)
    prepared, total, phases = [], [], []
    for repetition in range(repeats):
        start = perf_counter()
        context = setup()
        ready = perf_counter()
        value, info = evaluate(context)
        stopped = perf_counter()
        prepared.append(stopped - ready)
        total.append(stopped - start)
        phases.append(info)
        print(f"{label}, sample {repetition + 1}/{repeats}: "
              f"prepared={prepared[-1]:.6f}s total={total[-1]:.6f}s", flush=True)
    return value, summary(prepared), summary(total), phases


def main():
    """Run requested full solves, checking actual threads and accuracy before reporting."""
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--rigid-root", type=Path, default=rigid.DEFAULT_ROOT,
                        help=f"RigidMultiblobsWall checkout of branch {rigid.BRANCH} "
                             f"({RIGID_URL}) containing {RIGID_KERNEL} and {RIGID_DRIVER} "
                             "(default: %(default)s)")
    parser.add_argument("--sizes", type=int, nargs="+", default=[4096, 32768, 131072])
    parser.add_argument("--tolerances", type=float, nargs="+", default=[1e-6, 1e-8])
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--output", type=Path, default=root / "results/optimized_numba_comparison")
    args = parser.parse_args()
    if args.threads < 1 or args.repeats < 3 or any(n < 1 for n in args.sizes):
        parser.error("Use positive counts/threads and at least three repetitions")
    if any(not 0 < tol < 1 for tol in args.tolerances):
        parser.error("Tolerances must lie strictly between zero and one")
    rigid_root = rigid.check_root(args.rigid_root)
    rigid_checkout = rigid.describe_checkout(rigid_root)
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.update(OMP_NUM_THREADS=str(args.threads), NUMBA_NUM_THREADS=str(args.threads),
                      OMP_DYNAMIC="FALSE", OPENBLAS_NUM_THREADS="1", MKL_NUM_THREADS="1",
                      NUMBA_CACHE_DIR=str(args.output.resolve() / "numba-cache"))
    sys.path.insert(0, str(root / "python"))
    sys.path.insert(0, str(rigid_root))

    import numpy as np
    import numba
    import fmm3dpy
    import blake
    # External comparator: RigidMultiblobsWall's ``mobility`` package, not a local module.
    from mobility import blake_numba as optimized

    if Path(optimized.__file__).resolve() != rigid_root / RIGID_KERNEL:
        raise SystemExit(f"Imported {optimized.__file__}, not the RigidMultiblobsWall "
                         f"kernel under {rigid_root}; another 'mobility' package shadows it.")
    comparison = load_comparison(rigid_root / RIGID_DRIVER)

    def point_kernel(source, target, force, source_radius, target_radius, eta, periodic, wall):
        """Adapt the old validation signature strictly to the new point-force-only kernel."""
        assert wall == 1 and not np.any(source_radius) and not np.any(target_radius)
        assert not np.any(periodic)
        return optimized._point_force_kernel(source, target, force, eta)

    # Retain paths and runtime queries, without adding file fingerprints to reports.
    comparison.fingerprint = lambda path: {"path": str(Path(path).resolve())}
    openmp = comparison.inspect_fmm_openmp(Path(fmm3dpy.__file__).parent)
    rigid.require_fmm_threads(openmp, args.threads)
    if numba.get_num_threads() != args.threads:
        raise SystemExit(f"Numba reports {numba.get_num_threads()} threads, "
                         f"expected {args.threads}")
    optimized.blake_velocity()
    validation = comparison.validate(
        np, blake, point_kernel, SimpleNamespace(seed=927, tolerances=args.tolerances)
    )
    print(f"Validated specialized kernel and FMM; verified {args.threads} threads in "
          "Numba and both OpenMP extensions", flush=True)
    cpu = rigid.cpu_model()
    report = dict(
        status="inProgress", timestamp_utc=datetime.now(timezone.utc).isoformat(),
        settings=dict(sizes=args.sizes, tolerances=args.tolerances, threads=args.threads,
                      repeats=args.repeats, seed=927, reference_targets=64),
        geometry="independent unit-square lateral clouds; source heights [0.2,0.4]; "
                 "target heights [0.1,1.1]; wall=0; mu=1; no radii or periodic images",
        timing_scope="paired preparation/application; prepared excludes common geometry and "
                     "direct layout conversion, includes all direct kernel work or full FMM "
                     "guard/packing/two calls/assembly; imports/JIT/reference checks excluded",
        correction=False, diagnostics=False,
        provenance=dict(
            python=sys.version, executable=sys.executable, cpu=cpu,
            packages={name: importlib.metadata.version(name)
                      for name in ("numpy", "numba", "llvmlite", "fmm3dpy")},
            optimized_source=str(Path(optimized.__file__).resolve()),
            optimized_kernel=optimized._point_force_kernel.py_func.__name__,
            fmm_package=str(Path(fmm3dpy.__file__).resolve()),
            halfspace_package=str(Path(blake.__file__).resolve()),
            rigid_multiblobs_root=str(rigid_root), rigid_multiblobs_url=RIGID_URL,
            original_driver=str(rigid_root / RIGID_DRIVER), rigid_multiblobs=rigid_checkout,
            numba_threads=numba.get_num_threads(), numba_threading_layer=numba.threading_layer(),
            fmm_openmp=openmp,
            environment={key: os.environ.get(key) for key in
                         ("OMP_NUM_THREADS", "NUMBA_NUM_THREADS", "OMP_DYNAMIC",
                          "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS")},
        ), validation=validation, direct_cases=[], cases=[],
    )
    for case_index, n in enumerate(sorted(set(args.sizes))):
        seed = 1927 + case_index
        y, force, x = comparison.fixture(np, n, n, seed)

        def direct_setup():
            """Match the historical common geometry and point-major layout preparation."""
            geom = blake.prepare(y, x)
            return tuple(np.array(a.T, dtype=np.float64, order="C", copy=True)
                         for a in (geom.sources, geom.targets, force))

        def direct_apply(context):
            """Include contiguous component copies, all pairs, allocation and physical scaling."""
            return optimized._point_force_kernel(*context, 1.0).reshape(n, 3).T, {}

        direct, direct_time, direct_total, _ = timed_pairs(
            direct_setup, direct_apply, args.repeats, f"N={n} optimized Numba"
        )
        indices = np.unique(np.linspace(0, n - 1, min(64, n), dtype=int))
        geom = blake.prepare(y, x[:, indices])
        reference = blake.direct_velocity(geom, force)
        scale = blake.constituent_scale(geom, force)
        direct_error = comparison.error_metrics(np, blake, direct[:, indices], reference, scale)
        comparison.check_metrics(direct_error, normalized=5e-12, relative=2e-11)
        # Preserve expensive direct measurements even if a native FMM call is killed.
        report["direct_cases"].append(dict(
            ns=n, nt=n, seed=seed, direct_prepared=direct_time,
            direct_end_to_end=direct_total, direct_error=direct_error,
            reference_target_indices=indices.tolist(),
        ))
        comparison.write_json(args.output / "comparison.json", report)
        for tolerance in args.tolerances:

            def fmm_setup():
                """Include common geometry checks in paired end-to-end timing."""
                return blake.prepare(y, x)

            def fmm_apply(prepared):
                """Time the real default FMM path with correction and diagnostics disabled."""
                return blake.apply(prepared, force, method="fmm", tol=tolerance,
                                   correct_near_wall=False, diagnostics=False, return_info=True)

            value, fmm_time, fmm_total, phases = timed_pairs(
                fmm_setup, fmm_apply, args.repeats, f"N={n} FMM tol={tolerance:g}"
            )
            error = comparison.error_metrics(np, blake, value[:, indices], reference, scale)
            comparison.check_metrics(error, normalized=100*tolerance + 1e-12,
                                     relative=100*tolerance + 1e-11)
            assert all(info["fmm_call_count"] == 2 for info in phases)
            row = dict(ns=n, nt=n, seed=seed, tolerance=tolerance,
                       direct_prepared=direct_time, direct_end_to_end=direct_total,
                       fmm_prepared=fmm_time, fmm_end_to_end=fmm_total,
                       prepared_speedup=direct_time["median_seconds"] / fmm_time["median_seconds"],
                       end_to_end_speedup=direct_total["median_seconds"] / fmm_total["median_seconds"],
                       direct_error=direct_error, fmm_error=error,
                       fmm_vs_direct=comparison.field_difference(np, value, direct),
                       reference_target_indices=indices.tolist(), fmm_phases=phases)
            report["cases"].append(row)
            comparison.write_json(args.output / "comparison.json", report)
            print(f"N={n} tol={tolerance:g}: optimized={direct_time['median_seconds']:.6f}s "
                  f"FMM={fmm_time['median_seconds']:.6f}s speedup={row['prepared_speedup']:.3f}x; "
                  "accuracy passed", flush=True)
    report["status"] = "passed"
    comparison.write_json(args.output / "comparison.json", report)
    with (args.output / "comparison.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["ns", "nt", "tolerance", "direct_prepared", "fmm_prepared", "speedup"])
        for row in report["cases"]:
            writer.writerow([row["ns"], row["nt"], row["tolerance"],
                             row["direct_prepared"]["median_seconds"],
                             row["fmm_prepared"]["median_seconds"], row["prepared_speedup"]])


if __name__ == "__main__":
    main()
