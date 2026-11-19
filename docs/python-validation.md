# Python verification record

Measured on 2026-11-07 using the existing compiled `fmm3dpy` installation.
The accelerated implementation uses the representation of Z. Gimbutas,
L. Greengard and S. Veerapaneni; the comparison uses an independent Python
implementation of Blake's original tensor. See [usage and conventions](python.md).

## Pressure, velocity gradients and traction (2026-11-19)

Python now evaluates pressure, velocity gradients and traction with both
methods. See [usage](python.md#pressure-velocity-gradients-and-traction).

**Test suite.** All **33 Python acceptance tests passed**: the 22 existing
tests and 11 new tests in
[test_fields.py](../python/tests/test_fields.py). None were skipped or
incomplete. The run used `OMP_NUM_THREADS=1`, Python 3.10.12, NumPy 2.2.6,
SciPy 1.15.3 and the user-installed `fmm3dpy` 2.1.0. The new tests cover:

- **Blocked versus pair loops.** Blocked direct pressure and gradients agree
  with readable pair loops for three block sizes, including targets on the
  wall.
- **Finite-difference diagnostics** (in tests only). Finite differences of
  velocity converge to the analytic gradient at second order: error ratios
  are about 100 per decade, and relative errors fall to 1.5e-9 at step 1e-5.
  Differences of pressure and gradient satisfy `grad p = mu*laplacian(u)` to
  1e-6.
- **Physical identities.**
  - Velocity gradients are divergence-free.
  - At wall targets, tangential derivatives and `dw/dz` vanish, while the
    computed shear rates stay nonzero.
  - Traction matches the stress tensor applied to random unit normals.
  - Pressure and traction are independent of viscosity; gradients scale as
    `1/mu`.
  - Under a length scaling `s`, pressure and gradients scale as `1/s^2`.
  - Fields are unchanged when the wall and points are translated.
- **FMM on a real tree.** With 1,200 sources and 19 targets, five of them on
  the wall, `tol=1e-10` and `mu=1.7`, FMM velocity, pressure, gradients and
  traction agree with direct Blake to 1e-8 relative. The tests also check the
  FMM target flags.
- **Near-wall correction** gives bitwise-direct fields at the selected targets.
- **Errors and empty inputs.** Invalid requests and normals raise errors,
  coincident points raise `SingularInteractionError`, and empty inputs keep
  their output shapes.

**Cross-language check.** MATLAB `blake.fields` and Python `blake.fields`
were compared on identical inputs. The fixture has 1,500 sources at heights
0.05–1.05 and 40 targets: six on the wall, four at height `1e-3`, and the rest
at heights 0.02–1.02. It used `mu=1.7`, random unit normals and one thread.
Relative differences between the two languages:

| Method | Velocity | Pressure | Gradient | Traction |
|---|---:|---:|---:|---:|
| Direct Blake | 9.9e-16 | 1.1e-15 | 6.0e-16 | 1.2e-15 |
| FMM, `tol=1e-10` | 0 | 0 | 4.9e-17 | 1.9e-16 |

Python FMM accuracy against Python direct on the same fixture:

| FMM tolerance | Velocity | Pressure | Gradient | Traction |
|---|---:|---:|---:|---:|
| `1e-6` | 1.0e-6 | 6.1e-7 | 4.4e-7 | 7.7e-7 |
| `1e-10` | 2.5e-14 | 1.2e-15 | 4.1e-14 | 7.6e-14 |

At the six wall targets, the largest tangential derivative computed by direct
summation is 1.3e-14, against a largest gradient entry of 6.1e2. No MATLAB
files were changed, and no benchmarks were run for the new fields.

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

The validation command generates a JSON report for your installation. It
contains every measured field case, raw wall component maxima, both wrapper
call timings, direct and diagnostic timings, normalization checks, package
versions and resolved dependency paths. Undefined relative wall error is JSON null.

## Field geometry and accuracy

The readability refactor was rechecked on 2026-11-17: all 13 Python tests and
28 MATLAB tests passed. Pre/post comparison of the direct and FMM velocities,
pair tensors and constituent scales was bit-for-bit identical for three
deterministic geometries (seed 17011; Ns/Nt 1200/81, 73/41 and 13/19; base source
heights 0.2, 1e-6 and 0.01; the last case uses wall height 1.3 and viscosity 1.7).
The comparison used `tol=1e-12` and `OMP_NUM_THREADS=1` with the same dependencies
listed below. The complete field and reciprocity checks measured 2.111 s.
This is a correctness execution record, with no performance conclusion.

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

The measurements used Python 3.10.12, with NumPy
2.2.6 and `fmm3dpy` 2.1.0, on Linux x86-64, glibc 2.35, kernel
6.8.0-138-generic. The configured OpenMP thread count was one.
The inspected Python extension dynamic dependencies contain no separate
`libfmm3d.so` or `libgomp.so` entry; the `OMP_NUM_THREADS` setting alone does not
establish parallel capability of these binaries. No external FMM3D files were
modified or rebuilt.
