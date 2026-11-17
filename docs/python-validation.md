# Python verification record

Measured on 2026-11-07 using the existing compiled `fmm3dpy` installation.
The accelerated implementation uses the representation of Z. Gimbutas,
L. Greengard and S. Veerapaneni; the comparison uses an independent Python
implementation of Blake's original tensor. See [usage and conventions](python.md).

## Acceptance checks

- Python acceptance: **13 passed, 0 failed, 0 incomplete** in the final suite
  run. This includes real compiled Stokes and Laplace calls,
  normalization probes, field/wall/reciprocity checks, Cartesian forces,
  viscosity/linearity/scaling, duplicates, empty/singleton shapes, singularity
  errors, the FMM proximity guard, status propagation and malformed-output
  rejection.
- Missing dependency policy was separately exercised: full acceptance returned
  exit code 1 with five incomplete FMM tests; `--direct-only` returned 0 while
  retaining those five incomplete tests.
- The required MATLAB command completed successfully: **28 passed**, with no
  unavailable FMM tests. No MATLAB files were changed.
- Python syntax parsing passed for all ten new Python files; `git diff --check`
  passed. No benchmark profiles were run.

Reproduce from the repository root with an interpreter that imports the existing
NumPy and FMM3D packages:

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python3 python/tests/run_tests.py
PYTHONPATH=python OMP_NUM_THREADS=1 python3 examples/validate_velocity.py --output results/python_validation.json
OMP_NUM_THREADS=1 matlab -batch "addpath('matlab','tests'); blake.setup; run_tests;"
```

The generated, Git-ignored [JSON report](../results/python_validation.json)
contains every measured field case, raw wall component maxima, both wrapper
call timings, direct and diagnostic timings, normalization checks, package
versions and resolved dependency paths. Undefined relative wall error is JSON null.

## Field geometry and accuracy

Seed 927 generates signed normal random forces and independent lateral
coordinates in a unit square. For each geometry, source heights are in
`[h,2h]` above the wall. There are 13 wall targets directly below sources and
13 targets `1e-10` above the wall. Remaining bulk target heights lie in
`[0.1,1.1]`. The horizontally translated case adds `1e6` to both horizontal
coordinates. The shifted-wall case uses wall height 1.3 and viscosity 1.7;
other cases use wall height zero and viscosity one. Every source contributes
to every direct-reference target.

The table shows requested tolerance **1e-12**, with **OMP_NUM_THREADS=1**.
Absolute errors and wall residuals use maximum pointwise Euclidean norms.
Bulk relative errors use Frobenius norms. Wall normalized residuals divide by
the sum of the magnitudes of individual representation constituents before
cancellation. Wall values are actual computed values, without overwriting.

| Geometry | Ns / Nt | h | Bulk relative error | Wall absolute residual | Wall normalized residual |
|---|---:|---:|---:|---:|---:|
| Bulk and wall | 1200 / 81 | 0.2 | 8.444e-15 | 3.904e-14 | 6.120e-17 |
| Shallow sources | 73 / 41 | 1e-6 | 4.992e-11 | 1.783e-5 | 5.594e-11 |
| Shifted wall and viscosity | 73 / 41 | 0.01 | 1.998e-14 | 1.800e-13 | 2.853e-15 |
| Horizontal translation | 73 / 41 | 0.2 | 2.364e-15 | 2.616e-15 | 5.924e-17 |

These small fixtures test correctness. Their single-evaluation timings mainly
reflect the cost of invoking the FMM on small inputs and do not establish
large-system throughput, scaling or the crossover with direct summation.
Raw timings remain in the JSON report as an execution record. FMM time includes
geometry, guard, packing, both wrappers and assembly; direct time excludes the
separately timed optional constituent diagnostics.

A meaningful performance study would use warmed, repeated evaluations over
increasing source and target counts, with fixed geometry families, tolerances
and thread settings. Such a study has not been run here.

All 12 combinations of the four geometries and tolerances `1e-6`, `1e-9`,
`1e-12` passed the empirical normalized budget `100*tol+1e-12` and bulk-relative
budget `100*tol+1e-11`. The maximum normalized field error at `tol=1e-12`
was 5.594e-11 in the shallow case.

The shallow geometry has a tolerance-independent cancellation floor in this
installed backend: the wall absolute residual remains 1.783e-5 for all three
tolerances. The local singular constituent scale is large, so normalized and
absolute accuracy convey different information. Reducing `tol` cannot remove
this observed floor. The regular 1,200-source case changes from bulk relative
error 2.109e-6 at `tol=1e-6`, to 2.436e-10 at `tol=1e-9`, to 8.444e-15 at
`tol=1e-12`. Acceptance budgets describe these fixtures, not a guaranteed bound
on the relative error of a strongly cancelled half-space velocity.

## Reciprocity and normalization

Distributed reciprocity uses two distinct clouds with 1,200 and 701 sources,
lateral coordinates in the unit square, respective heights `[0.2,1.2]` and
`[0.3,1.3]`, viscosity one and wall height zero. Compare `sum(f_B*u_A)` with
`sum(f_A*u_B)`, normalized by `sum(abs(f_B*u_A))+sum(abs(f_A*u_B))`.
Pair reciprocity reconstructs all three force columns and compares the tensor
with the transpose after swapping source and target.

| Requested tolerance | Distributed FMM defect | Pair FMM defect |
|---:|---:|---:|
| 1e-6 | 4.543e-10 | 2.043e-16 |
| 1e-9 | 7.695e-17 | 2.043e-16 |
| 1e-12 | 7.695e-17 | 2.043e-16 |

The independent direct distributed defect was 4.965e-18.
The direct tensor also passed pair reciprocity tests over several heights,
viscosities and translated walls.

Wrapper normalization probes use seed 93, 1,200 sources in the unit cube and
31 targets in `[1.2,2.2]^3`, requested tolerance `1e-12` and the same thread
setting. Direct probes use one target workspace, without full interaction
storage. Relative Stokes error was 1.069e-13; the largest charge/dipole
potential or gradient relative error was 1.871e-12. Charges, dipoles and their
combination were tested separately.

## Dependency provenance

The measured interpreter was `/usr/bin/python3`, Python 3.10.12, with NumPy
2.2.6 and `fmm3dpy` 2.1.0, on Linux x86-64, glibc 2.35, kernel
6.8.0-138-generic. The configured OpenMP thread count was one.
The package resolved to
`/home/anna/.local/lib/python3.10/site-packages/fmm3dpy`.

The external FMM3D checkout reported revision
`1e73a3b595589bff25a7ab711f14af44870261c7`. This is an inspected checkout
revision, not proof that it produced the installed Python extensions. Their inspected
dynamic dependencies contain no separate `libfmm3d.so` or `libgomp.so` entry;
the `OMP_NUM_THREADS` setting alone does not establish parallel capability of
these binaries. No external FMM3D files were modified or rebuilt.
