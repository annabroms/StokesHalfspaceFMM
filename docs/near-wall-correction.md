# Near-wall correction in MATLAB and Python

Enable `correct_near_wall` in either interface:

```matlab
opts = struct('tol',1e-12,'correct_near_wall',true);
u = blake.velocity(sources,forces,targets,opts);
```

```python
u = blake.velocity(sources, forces, targets, tol=1e-12, correct_near_wall=True)
```

The default remains disabled. Selected targets use the complete independent
direct Blake sum over all sources; the other targets use the usual two FMM
calls. When all targets are selected, the calculation is entirely direct.
Wall residuals are always computed, never set to zero.
MATLAB also supports its optional one-call backend for the remaining targets,
and corrects all requested fields through the independent Blake references.

The [selection criterion and timing fields](python.md#optional-near-wall-correction)
are documented with the Python interface. Selection depends on nearest-source
distance relative to geometry size and the requested tolerance, rather than
target height alone. It can also select very close bulk targets. It is a
roundoff heuristic, not an error estimator or a guarantee of relative accuracy.
The accelerated representation remains that of Z. Gimbutas, L. Greengard and
S. Veerapaneni; correction uses Blake's original tensor independently.

## Choosing the cutoff

Use the automatic rule first: `d/L <= max(64*eps,8*eps/tol)`, capped at 2.
Here `d` is nearest-source distance, `L` includes physical sources, images and
targets, and `eps` is float64 machine epsilon. At `tol=1e-12` the cutoff is
approximately `1.8e-3*L`; at `tol=1e-9` it is `1.8e-6*L`. The latter can leave
larger absolute wall slip while still meeting a much looser conditioning goal.

For high-accuracy wall values, **`1e-3*L` is a useful explicit starting point**.
Use `1e-4*L` if reducing direct work matters more and the resulting wall residual
is acceptable. Set the positive `near_wall_distance` option in coordinate units
in either language. Its default is `[]` in MATLAB or `None` in Python, meaning
automatic selection. For example, with prepared geometry:

```matlab
opts.near_wall_distance = 1e-3*geom.span;
```

```python
u = blake.apply(geom, forces, correct_near_wall=True,
                near_wall_distance=1e-3*geom.span)
```

A cutoff study used 2,048 sources and 1,024 targets in the unit square, source
heights log-uniform over `[1e-7,1e-2]` except one at `0.8`, 256 wall targets and
256 targets at height `1e-10`, each directly beneath a source, plus 512 bulk
targets at heights `[0.1,1.1]`. Forces were signed Gaussian samples, seed 171140,
`mu=1`, wall at zero, `tol=1e-12`, and one configured OpenMP thread. These are
Python accuracy measurements, not new runtime benchmarks.

| Cutoff / L | Corrected targets | Maximum wall velocity norm |
|---|---:|---:|
| Disabled | 0 | 3.54e-3 |
| 1e-6 | 117 | 2.57e-5 |
| 1e-5 | 213 | 2.25e-7 |
| 1e-4 | 323 | 3.09e-9 |
| 1e-3 | 431 | 4.46e-10 |
| Automatic, about 1.8e-3 | 461 | 4.46e-10 |

The independent direct sum itself has a maximum wall residual of `4.46e-10`
in this fixture. Increasing the cutoff beyond `1e-3*L` did not reduce that wall
floor, although near-wall relative agreement improved from `2.09e-12` to
`8.40e-13`. Thus `1e-3*L` was a useful tradeoff here, while the default automatic
rule was more conservative. No single cutoff guarantees absolute wall accuracy
for all source heights, force scales, viscosities or geometries. Check both
the residual and `info.corrected_target_count`: every selected target entails
a complete O(Ns) direct sum.

## Python accuracy

Measured on 2026-11-17, with `tol=1e-12`, `mu=1`, wall at zero and
`OMP_NUM_THREADS=1`. The shallow fixture has 73 sources with heights in
`[1e-6,2e-6]` and 41 targets: 13 on the wall directly below sources,
13 at height `1e-10`, and 15 in the bulk. Horizontal coordinates lie in the unit
square. Thirteen targets are selected.

| Quantity | Flag off | Flag on |
|---|---:|---:|
| Maximum wall velocity norm | 1.7835e-5 | 3.8197e-11 |
| Maximum error against direct Blake at wall targets | 1.7835e-5 | 0 |
| Maximum absolute error at the other near-wall targets | 6.2339e-14 | 6.2339e-14 |
| Bulk relative error against direct Blake | 4.9917e-11 | 4.9917e-11 |

Velocity errors retain physical units: m/s for SI inputs. The corrected wall
residual is about **470,000 times smaller**. Agreement with the direct evaluator
at corrected targets is expected because that evaluator supplies their values.
Separate tests compare its contractions with independently assembled pair
tensors. The nonzero wall residual measures remaining direct-sum roundoff.

Near-wall targets far horizontally from a source need no correction here:
their absolute errors are already about `6e-14`. Their small velocities make
the relative error about 1.4%, illustrating why relative error alone can be
misleading near a no-slip wall. Relative wall error is undefined. Use absolute
wall slip and constituent-normalized error alongside relative error where the
reference velocity is appreciably nonzero.

In the 8,192-source/8,192-target case with 32 close targets, maximum wall slip
decreased from `8.1698e-5` to `3.8193e-11`. For the sampled close targets at
height `1e-10`, relative error against direct Blake decreased from `8.87e-3`
to agreement with the direct sum. The bulk relative error remained `8.24e-11`.
For 32,768 sources and targets with 32 close targets, maximum wall slip fell
from `1.0042e-4` to `2.0721e-11`; close near-wall relative error fell from
`1.19e-2` to agreement with the direct sum, while bulk relative error remained
`7.50e-12`.

Reciprocity was also tested on unequal clouds of 1,200 and 701 points. The
first cloud has source heights `[1e-6,2e-6]`; the second contains 13 points
at height `1e-10` below corresponding points and 688 bulk points at heights
`[0.2,1.2]`. Independent signed Gaussian forces are used on each cloud. The
defect is `|fB·uA - fA·uB| / (sum|fB*uA| + sum|fA*uB|)`.

| Reciprocity defect | Value |
|---|---:|
| Flag off | 1.4190e-3 |
| Flag on (13 and 14 targets corrected in the two directions) | 8.1946e-11 |
| Independent direct evaluation in both directions | 2.9146e-10 |

This strongly cancelling test reaches the numerical accuracy of direct
evaluation, rather than the requested `1e-12`. The smaller corrected defect
than the direct defect reflects cancellation of numerical errors; it does not
establish greater accuracy than direct summation.

## MATLAB accuracy

The MATLAB two-call path was checked on 1,200 sources and 67 targets using
MATLAB R2025a, the existing compiled FMM3D MATLAB wrappers, `tol=1e-12`, `mu=1`
and `OMP_NUM_THREADS=1`. Sources have unit-square horizontal coordinates and
heights `[1e-6,2e-6]`, except one at `0.8`. Thirteen targets lie on the wall
under sources, another thirteen at height `1e-10` under sources, and 41 in the
bulk at heights `[0.1,1.1]`. The wall is at zero and the MATLAB seed is 171139.

Twenty-six targets were corrected. Maximum wall velocity norm decreased from
`5.5014e-5` to `4.1431e-11`, equal to the independently evaluated direct-sum
wall residual. Bulk relative error against direct Blake was `2.6840e-11`.
The other targets retained two FMM calls. The recorded accuracy check includes
individual phase runtimes and resolved wrapper/extension provenance; these
small-case single-evaluation times are not used as performance evidence.

## Speed at cutoff 1e-3*L

Measured on 2026-11-17 with the **explicit** cutoff `near_wall_distance=1e-3*L`,
`tol=1e-12`, `mu=1`, wall at zero and the default **two-call FMM backend** in
both languages. Here `L` is the maximum side length of the bounding box of
sources, images and targets. It was approximately `1.9` in all four cases,
so the physical cutoff was approximately `0.0019`.

Both interfaces received identical exported arrays. Runs were sequential,
with `OMP_NUM_THREADS=1`; MATLAB also used `-singleCompThread`. Each path was
warmed once, then timed three times with alternating off/on order. The table
reports medians of complete public velocity calls, including geometry
preparation, validation, exclusion checks, selection, strength packing, both
FMM calls, assembly and direct correction. Reference comparisons and optional
diagnostics were outside the timed region. Trees were rebuilt on every call.

| Interface | Sources / targets | Corrected when on | Off, s | On, s | Time change |
|---|---|---:|---:|---:|---:|
| MATLAB | 8,192 / 8,192, few close | 32 | 7.002 | 7.163 | +2.3% |
| MATLAB | 8,192 / 8,192, quarter close | 2,048 | 7.296 | 9.878 | +35.4% |
| MATLAB | 8,192 / 8,192, varied heights | 3,545 | 7.184 | 11.644 | +62.1% |
| MATLAB | 32,768 / 32,768, few close | 32 | 19.459 | 19.123 | -1.7% |
| Python | 8,192 / 8,192, few close | 32 | 8.005 | 8.138 | +1.7% |
| Python | 8,192 / 8,192, quarter close | 2,048 | 8.839 | 10.463 | +18.4% |
| Python | 8,192 / 8,192, varied heights | 3,545 | 8.104 | 12.097 | +49.3% |
| Python | 32,768 / 32,768, few close | 32 | 21.410 | 20.390 | -4.8% |

**Correction remains off by default in both languages.** Sparse correction
was inexpensive in these cases, but correcting 25% or 43.3% of the targets
increased runtime substantially. Every corrected target requires a complete
O(Ns) direct sum in bounded blocks. Removed targets reduce the FMM workload,
which can offset some correction cost; the modest gains on the largest sparse
case should not be assumed for other geometries. Three repetitions establish
these local measurements, not a general speed guarantee.

The measured direct correction cost was:

| Corrected targets / sources | MATLAB direct work, s | Python direct work, s |
|---|---:|---:|
| 32 / 8,192 | 0.048 | 0.035 |
| 2,048 / 8,192 | 2.700 | 2.305 |
| 3,545 / 8,192 | 4.481 | 3.900 |
| 32 / 32,768 | 0.157 | 0.134 |

These are medians of the separately measured direct phase, so they are not
obtained by subtracting median total times. Selection itself took about
`0.07–0.09 s` at 8,192 targets and `0.28–0.30 s` at 32,768 targets.

For the few-close and quarter-close cases, source heights were uniform in
`[1e-6,2e-6]`, except one source at `0.8`. Half the designated close targets
were on the wall and half at height `1e-10`, directly beneath source positions.
The varied-height case had source heights log-uniform in `[1e-7,1e-2]`, except
one at `0.8`, and 4,096 designated close targets; the cutoff selected 3,545.
Remaining targets had heights `[0.1,1.1]`. All horizontal coordinates occupied
a translated unit square, with signed Gaussian forces. The seeds were
`171138 + Ns + designated_close_count` for uniform-height cases and `171141`
for the varied-height case.

Accuracy was checked outside timing. In the varied-height case, maximum wall
slip fell from `2.93e-2` to `6.11e-10` in both languages. Correction retained
all target fields and never assigned zero to wall residuals.

## Environment and reproduction

The measurements used the existing compiled FMM3D interfaces: MATLAB R2025a,
`fmm3dpy 2.1.0`, NumPy `2.2.6` and Python `3.10.12`, on Linux x86-64 with one
configured OpenMP thread. The paired timing study also fixed MATLAB computational
threads to one. No external FMM3D code was modified. Package versions identify
the installed interfaces without asserting correspondence to an unverified
upstream source revision.

From the repository root, using the interpreter containing these dependencies,
prepare shared inputs once and run the paired speed study sequentially:

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python3 examples/compare_near_wall_rule.py --prepare-only
OMP_NUM_THREADS=1 matlab -singleCompThread -batch "addpath('matlab','examples'); blake.setup; compare_near_wall_rule(3);"
PYTHONPATH=python OMP_NUM_THREADS=1 python3 examples/compare_near_wall_rule.py --repetitions 3
```

The original automatic-cutoff comparison, distance sensitivity study and
acceptance suites remain independently runnable:

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python3 examples/compare_near_wall.py
PYTHONPATH=python OMP_NUM_THREADS=1 python3 examples/compare_near_wall.py --distance-sweep
PYTHONPATH=python OMP_NUM_THREADS=1 python3 python/tests/run_tests.py
OMP_NUM_THREADS=1 matlab -batch "addpath('matlab','tests'); blake.setup; run_tests;"
```

The comparison saves strict JSON with raw repeated timings, phase timings,
sampled errors, wall residuals and dependency provenance. Reference samples
include up to 32 points in each of the wall, near-wall and bulk groups, always
summed over every source; wall slip is measured over every wall target.
The small accuracy fixture uses seed 927 after generating the preceding bulk
fixture, and then the same random stream for reciprocity.

Acceptance after implementation: **19 Python tests and 37 MATLAB tests passed**,
with no FMM tests skipped. Checks cover independent pair tensors, computed
wall residuals, mixed-path reciprocity, shifts and viscosity, selection under
coordinate scaling, exact coincidence errors, unavailable dependencies and the
two-call contract for remaining FMM targets. Python syntax and whitespace
checks also passed. MATLAB Code Analyzer reported no issues on changed files.
MATLAB checks include both FMM compositions, the velocity-only Laplace adapter,
mixed direct/FMM pressure, gradients and traction, a 1,200-source multipole
fixture, distributed reciprocity on unequal clouds and explicit-distance
selection under coordinate scaling. Both languages use the same portable
selection fixture and expected masks.
