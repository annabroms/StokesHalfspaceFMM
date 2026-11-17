"""Measure flag off/on at distance=1e-3*span on common MATLAB/Python fixtures."""

import argparse
import json
from pathlib import Path
from time import perf_counter

import numpy as np

import blake
from blake.near_wall import select_targets
from blake.validation import provenance
from compare_near_wall import clean_json, geometry, sampled_accuracy


def fixtures():
    """Create linear-storage common inputs covering sparse and frequent correction."""
    cases = []
    for label, ns, nt, close_count, varied in (
        ("few close", 8192, 8192, 32, False),
        ("quarter close", 8192, 8192, 2048, False),
        ("varied heights", 8192, 8192, 4096, True),
        ("larger few close", 32768, 32768, 32, False),
    ):
        geom, forces, wall_count = geometry(ns, nt, close_count)
        seed = 171138 + ns + close_count
        if varied:
            seed = 171141
            rng = np.random.default_rng(seed)
            sources = np.vstack((rng.random((2, ns)), 10 ** rng.uniform(-7, -2, ns)))
            sources[2, -1] = 0.8
            targets = np.vstack((rng.random((2, nt)), 0.1 + rng.random(nt)))
            targets[:2, :close_count] = sources[:2, :close_count]
            targets[2, :wall_count] = 0
            targets[2, wall_count:close_count] = 1e-10
            geom = blake.prepare(sources, targets)
            forces = rng.normal(size=sources.shape)
        # These exported coordinates are identical inputs to both public wrappers.
        geom = blake.prepare(geom.sources, geom.targets)
        cutoff = 1e-3 * geom.span
        selected, _ = select_targets(geom, tol=1e-12, distance=cutoff)
        cases.append({
            "case": label, "ns": ns, "nt": nt, "seed": seed,
            "sources": geom.sources.tolist(), "targets": geom.targets.tolist(),
            "forces": forces.tolist(), "span": geom.span, "near_wall_distance": cutoff,
            "expected_corrected_targets": int(selected.sum()),
            "close_targets": close_count, "wall_targets": wall_count,
            "source_heights": "log-uniform [1e-7,1e-2], except one at 0.8" if varied
                              else "uniform [1e-6,2e-6], except one at 0.8",
            "horizontal_domain": "unit square, translated horizontally before export",
            "bulk_target_heights": "[0.1,1.1]", "close_near_wall_height": 1e-10,
        })
    return {"tol": 1e-12, "mu": 1.0, "wall_height": 0.0,
            "relative_cutoff": 1e-3, "cases": cases}


def compare_case(case, repetitions):
    """Time complete velocity calls after warming, with reference checks outside timing."""
    sources = np.asarray(case["sources"])
    targets = np.asarray(case["targets"])
    forces = np.asarray(case["forces"])
    options = dict(tol=1e-12, near_wall_distance=case["near_wall_distance"])
    fields, samples, phases = {}, {False: [], True: []}, {False: [], True: []}
    print(f"Python: starting {case['case']} ({case['ns']}/{case['nt']})", flush=True)
    for flag in (False, True):
        blake.velocity(sources, forces, targets, correct_near_wall=flag, **options)
    for repetition in range(repetitions):
        for flag in ((False, True) if repetition % 2 == 0 else (True, False)):
            start = perf_counter()
            fields[flag], info = blake.velocity(
                sources, forces, targets, correct_near_wall=flag,
                return_info=True, **options,
            )
            samples[flag].append(perf_counter() - start)
            phases[flag].append(info)
            expected = case["expected_corrected_targets"] if flag else 0
            if info["corrected_target_count"] != expected:
                raise AssertionError("Timed evaluation did not use the expected cutoff")
        print(f"Python: {case['case']}, repetition {repetition + 1}/{repetitions}: "
              f"off={samples[False][-1]:.3f}s, on={samples[True][-1]:.3f}s", flush=True)
    before, after = (float(np.median(samples[flag])) for flag in (False, True))
    geom = blake.prepare(sources, targets)
    accuracy = sampled_accuracy(geom, forces, fields, case["wall_targets"], case["close_targets"])
    result = {key: value for key, value in case.items()
              if key not in ("sources", "forces", "targets")}
    result.update({
        "off_seconds": before, "on_seconds": after, "on_over_off": after / before,
        "raw_off_seconds": samples[False], "raw_on_seconds": samples[True],
        "off_phases": phases[False], "on_phases": phases[True], "accuracy": accuracy,
    })
    print(f"Python: {case['case']}, corrected={case['expected_corrected_targets']}, "
          f"off={before:.4f}s, on={after:.4f}s, ratio={after / before:.3f}", flush=True)
    return result


def main():
    """Export shared fixtures or run only the explicitly requested Python comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--repetitions", type=int, default=3)
    args = parser.parse_args()
    if args.repetitions < 3:
        parser.error("Use at least three repetitions")
    folder = Path(__file__).resolve().parents[1] / "results"
    folder.mkdir(exist_ok=True)
    fixture_path = folder / "near_wall_rule_fixtures.json"
    if args.prepare_only:
        fixture_path.write_text(json.dumps(fixtures(), allow_nan=False))
        print("Prepared four identical-input MATLAB/Python fixtures", flush=True)
        return
    if not fixture_path.exists():
        parser.error("Run with --prepare-only first to create shared inputs")
    data = json.loads(fixture_path.read_text())
    report = {"provenance": provenance(), "repetitions": args.repetitions,
              "relative_cutoff": data["relative_cutoff"], "tol": data["tol"],
              "mu": data["mu"], "wall_height": data["wall_height"],
              "timing_scope": "complete public velocity call including geometry, exclusion guard, "
                              "selection, packing, two FMM calls, assembly and direct correction; "
                              "one warm-up per path; alternating order; reference checks excluded",
              "backend": "default two-call", "cases": []}
    for case in data["cases"]:
        report["cases"].append(compare_case(case, args.repetitions))
        (folder / "near_wall_rule_python.json").write_text(
            json.dumps(clean_json(report), indent=2, allow_nan=False) + "\n"
        )


if __name__ == "__main__":
    main()
