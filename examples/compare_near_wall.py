"""Compare warmed Python FMM evaluations with optional direct target replacement."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import blake
from blake.near_wall import select_targets, target_subset
from blake.validation import provenance


def geometry(source_count, target_count, close_count, *, shallow=True):
    """Make reproducible unit-square clouds with specified close wall/near-wall targets."""
    rng = np.random.default_rng(171138 + source_count + close_count)
    sources = np.vstack((rng.random((2, source_count)), rng.random(source_count)))
    sources[2] = (1e-6 if shallow else 0.2) * (1 + sources[2])
    sources[2, -1] = 0.8
    targets = np.vstack((rng.random((2, target_count)), 0.1 + rng.random(target_count)))
    indices = np.arange(close_count) % (source_count - 1)
    targets[:, :close_count] = sources[:, indices]
    wall_count = close_count // 2
    targets[2, :wall_count] = 0.0
    targets[2, wall_count:close_count] = 1e-10
    return blake.prepare(sources, targets), rng.normal(size=sources.shape), wall_count


def sampled_accuracy(geom, forces, fields, wall_count, close_count):
    """Compare stratified target samples against every source, outside the timed region."""
    target_count = geom.targets.shape[1]
    groups = {
        "wall": np.arange(wall_count),
        "near_wall": np.arange(wall_count, close_count),
        "bulk": np.arange(close_count, target_count),
    }
    report = {}
    for name, indices in groups.items():
        if not indices.size:
            continue
        indices = indices[np.linspace(0, indices.size - 1, min(32, indices.size), dtype=int)]
        subset = target_subset(geom, indices)
        reference = blake.direct_velocity(subset, forces)
        scale = blake.constituent_scale(subset, forces)
        report[name] = {
            "reference_target_count": indices.size,
            "reference_norm": float(np.linalg.norm(reference)),
            "off": blake.metrics(fields[False][:, indices], reference, scale),
            "on": blake.metrics(fields[True][:, indices], reference, scale),
        }
    if wall_count:
        report["wall_residual"] = {
            label: float(np.max(np.linalg.norm(fields[flag][:, :wall_count], axis=0)))
            for flag, label in ((False, "off"), (True, "on"))
        }
    return report


def compare_case(label, ns, nt, close_count, repetitions, tol, *, shallow=True):
    """Warm both paths, alternate timed order and retain samples and accuracy evidence."""
    geom, forces, wall_count = geometry(ns, nt, close_count, shallow=shallow)
    selected, radius = select_targets(geom, tol=tol)
    fields, samples, phases = {}, {False: [], True: []}, {False: [], True: []}
    for flag in (False, True):
        blake.apply(geom, forces, tol=tol, correct_near_wall=flag)
    for repetition in range(repetitions):
        order = (False, True) if repetition % 2 == 0 else (True, False)
        for flag in order:
            start = perf_counter()
            fields[flag], info = blake.apply(
                geom, forces, tol=tol, correct_near_wall=flag, return_info=True
            )
            samples[flag].append(perf_counter() - start)
            phases[flag].append(info)
    result = {
        "case": label, "ns": ns, "nt": nt, "tol": tol, "mu": 1.0,
        "wall_height": 0.0, "horizontal_domain": "unit square",
        "source_heights": "[1e-6,2e-6], except one at 0.8" if shallow
                          else "[0.2,0.4], except one at 0.8",
        "bulk_target_heights": "[0.1,1.1]", "near_wall_height": 1e-10,
        "close_targets": close_count, "wall_targets": wall_count,
        "corrected_target_count": int(selected.sum()), "relative_radius": radius,
        "median_seconds": {str(flag): float(np.median(samples[flag])) for flag in samples},
        "raw_seconds": {str(flag): samples[flag] for flag in samples},
        "phases": {str(flag): phases[flag] for flag in phases},
        "accuracy": sampled_accuracy(geom, forces, fields, wall_count, close_count),
    }
    before, after = result["median_seconds"]["False"], result["median_seconds"]["True"]
    print(f"{label}: {ns}/{nt}, corrected={selected.sum()}, "
          f"off={before:.4f}s, on={after:.4f}s, ratio={after / before:.3f}", flush=True)
    return result


def small_accuracy():
    """Reproduce the shallow-source accuracy fixture and measure corrected reciprocity."""
    rng = np.random.default_rng(927)
    for ns, nt, height in ((1200, 81, 0.2), (73, 41, 1e-6)):
        sources = np.vstack((rng.random((2, ns)), height * (1 + rng.random(ns))))
        targets = np.vstack((rng.random((2, nt)), 0.1 + rng.random(nt)))
        targets[2, :13] = 0
        targets[2, 13:26] = 1e-10
        targets[:2, :13] = sources[:2, :13]
        forces = rng.standard_normal((3, ns))
    geom = blake.prepare(sources, targets)
    fields = {}
    for flag in (False, True):
        fields[flag] = blake.apply(geom, forces, tol=1e-12, correct_near_wall=flag)
    report = sampled_accuracy(geom, forces, fields, 13, 26)
    report["ns"], report["nt"], report["tol"] = 73, 41, 1e-12
    report["corrected_target_count"] = int(select_targets(geom, tol=1e-12)[0].sum())

    # Unequal clouds, both above the wall, with close and distant target pairs.
    a = np.vstack((rng.random((2, 1200)), 1e-6 * (1 + rng.random(1200))))
    b = np.vstack((rng.random((2, 701)), 0.2 + rng.random(701)))
    b[:, :13] = a[:, :13]
    b[2, :13] = 1e-10
    fa, fb = rng.normal(size=a.shape), rng.normal(size=b.shape)
    report["reciprocity"] = {}
    for name, options in (("off", {}), ("on", {"correct_near_wall": True}),
                          ("direct", {"method": "blake"})):
        ab, ia = blake.velocity(a, fa, b, tol=1e-12, return_info=True, **options)
        ba, ib = blake.velocity(b, fb, a, tol=1e-12, return_info=True, **options)
        left, right = fb * ab, fa * ba
        report["reciprocity"][name] = {
            "defect": float(abs(left.sum() - right.sum()) /
                            (np.abs(left).sum() + np.abs(right).sum())),
            "corrected_targets": [ia["corrected_target_count"], ib["corrected_target_count"]],
            "cloud_counts": [1200, 701],
        }
    return report


def distance_sweep():
    """Measure cutoff sensitivity on varying source heights, without speed claims."""
    rng = np.random.default_rng(171140)
    ns, nt, close_count = 2048, 1024, 512
    sources = np.vstack((rng.random((2, ns)), 10 ** rng.uniform(-7, -2, ns)))
    sources[2, -1] = 0.8
    targets = np.vstack((rng.random((2, nt)), 0.1 + rng.random(nt)))
    targets[:2, :close_count] = sources[:2, :close_count]
    wall_count = close_count // 2
    targets[2, :wall_count] = 0.0
    targets[2, wall_count:close_count] = 1e-10
    forces = rng.normal(size=sources.shape)
    geom = blake.prepare(sources, targets)
    reference = blake.direct_velocity(geom, forces)
    scale = blake.constituent_scale(geom, forces)
    cases = []
    for relative_distance in (None, 1e-6, 1e-5, 1e-4, 1e-3, "auto"):
        enabled = relative_distance is not None
        distance = (None if relative_distance in (None, "auto")
                    else relative_distance * geom.span)
        value, info = blake.apply(
            geom, forces, tol=1e-12, correct_near_wall=enabled,
            near_wall_distance=distance, return_info=True,
        )
        case = {
            "cutoff": "off" if not enabled else relative_distance,
            "corrected_targets": info["corrected_target_count"],
            "effective_distance_over_span": info["near_wall_relative_radius"],
            "max_wall_slip": float(np.max(np.linalg.norm(value[:, :wall_count], axis=0))),
            "wall_error_vs_direct": blake.metrics(
                value[:, :wall_count], reference[:, :wall_count], scale[:wall_count]),
            "near_wall_error_vs_direct": blake.metrics(
                value[:, wall_count:close_count], reference[:, wall_count:close_count],
                scale[wall_count:close_count]),
        }
        cases.append(case)
        print(f"cutoff={case['cutoff']}, corrected={case['corrected_targets']}, "
              f"wall slip={case['max_wall_slip']:.3e}", flush=True)
    return {
        "provenance": provenance(), "seed": 171140, "ns": ns, "nt": nt,
        "tol": 1e-12, "mu": 1.0, "wall_height": 0.0,
        "horizontal_domain": "unit square", "span": geom.span,
        "source_heights": "log-uniform [1e-7,1e-2], except one at 0.8",
        "wall_targets": 256, "close_near_wall_targets": 256,
        "near_wall_height": 1e-10, "bulk_target_heights": "[0.1,1.1]",
        "reference_max_wall_slip": float(np.max(np.linalg.norm(reference[:, :wall_count], axis=0))),
        "note": "Accuracy and selected-count study; single evaluations are not speed measurements.",
        "cases": cases,
    }


def clean_json(value):
    """Convert NumPy counts and undefined wall relative errors to strict JSON values."""
    if isinstance(value, dict):
        return {key: clean_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [clean_json(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def main():
    """Run only the explicitly requested focused speed and accuracy comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, help="optional JSON report destination")
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--distance-sweep", action="store_true",
                        help="only measure cutoff sensitivity and selected counts")
    args = parser.parse_args()
    if args.output is None:
        args.output = Path("results/near_wall_distance.json" if args.distance_sweep
                           else "results/near_wall_comparison.json")
    if args.distance_sweep:
        report = distance_sweep()
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")
        return
    if args.repetitions < 3:
        parser.error("Use at least three timed repetitions")
    report = {"provenance": provenance(), "repetitions": args.repetitions,
              "timing_scope": "prepared geometry, selection, guard, both FMM calls and direct correction; "
                              "one warm-up per path; alternating order; accuracy work excluded",
              "small_accuracy": small_accuracy(), "cases": []}
    for label, ns, nt, close_count, shallow in (
        ("regular", 8192, 8192, 0, False),
        ("few close", 8192, 8192, 32, True),
        ("quarter close", 8192, 8192, 2048, True),
        ("all close", 8192, 8192, 8192, True),
        ("larger few close", 32768, 32768, 32, True),
    ):
        report["cases"].append(compare_case(
            label, ns, nt, close_count, args.repetitions, 1e-12, shallow=shallow
        ))
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
