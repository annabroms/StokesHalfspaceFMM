# Benchmarks and timing conventions

[Back to the main README](../README.md)

## Benchmarks

Launch a fresh MATLAB process for each thread setting, for example:

```bash
OMP_NUM_THREADS=1 matlab -batch "addpath('matlab','benchmarks'); blake.setup; run_benchmarks(struct('profile','quick'));"
OMP_NUM_THREADS=8 matlab -batch "addpath('matlab','benchmarks'); blake.setup; run_benchmarks(struct('profile','quick'));"
```

Run commands from this repository. Set `profile` to `default` for Ns=2^(5:2:17),
or `large` for Ns=2^(5:2:21). The quick profile stops at 8192. All profiles use
Nt/Ns=1/4,1,4, with wall targets included in Nt; source layers have base heights
1e-5,1e-2,1. The large profile can require substantial memory and runtime.

Drivers warm both interfaces, use five repetitions and median timings, record
hardware/software/thread/binary provenance, and save MAT, CSV and PNG/PDF outputs
under ignored `results/`. Configuration overrides include `sizes`, `ratios`,
`heights`, `repetitions`, `tol`, `subset_size`, `full_pair_limit`, `seed`, `save`,
and `stokes_backend`.

Above 2e7 source/target pairs, independent direct validation uses up to 256
stratified targets per category, summing **all sources**. Such direct timings are
subset timings; no full speedup is inferred. Crossover and slopes are descriptive
measurements. The tolerance sweep and block-size study use fixed moderate fixtures.
The source-loop alternative is represented by source block size 1.

## Timing schema and free-space backend comparison

New reports use `timing_schema_version=2`. Each FMM repetition executes once,
collecting component and total times from that same invocation. Total time
includes option validation, geometry, guard search, strength preparation,
backend calls and output assembly. `fmm_evaluation_seconds` sums the Stokes and
harmonic backend calls. `overhead_seconds` is their complement in total time;
for direct methods it is NaN. Packing/checks/dispatch are included in overhead.
Strength preparation, guard, geometry and assembly phases do not overlap.
With diagnostics enabled, its direct cost is included in total/overhead and
reported separately.

MAT/CSV sample tables preserve every repetition. Summary FMM time and overhead
are medians of their per-repetition values, not sums/differences of medians.
The old `prepared_total_seconds` and `fmm_apply_seconds` columns are retired.
Existing saved reports are unchanged and still work with `plot_benchmarks`.

The standalone comparison is a performance test, excluded from `run_tests`:

```matlab
setpath;
comparison = compare_stokes_backends;
% Optional configuration overrides:
% compare_stokes_backends(struct('sizes',[512 2048],'tol',1e-9,'save',false));
```

It compares the updated Fortran Stokes composition (`upstream`) with the
legacy MATLAB four-potential reference (`laplace-velocity`). It evaluates
**free-space Stokeslet velocity only**, using Ns ordinary sources
and independent targets; there is no wall, source reflection or harmonic
correction. Defaults are Ns=512,2048; Nt/Ns=1/4,1,4; seven paired repetitions;
a centered unit cube; fixed seed; mu=1; tolerance 1e-9. It warms both paths,
alternates execution order, and validates both against blocked direct Stokeslets
outside the timed sections. Above 2e7 pairs, up to 256 representative targets
are checked against all sources. It reports absolute, global relative and
constituent-normalized errors, backend differences, call and total timing ratios,
and absolute timing differences. No assertion requires a speedup. MAT/CSV files
save raw samples, summaries, configuration and provenance under unique names.
Run separate MATLAB processes for different thread settings; the driver does not
change thread settings. This comparison and benchmark profiles are not run by
the correctness suite.

## Performance compared with direct Numba evaluation

### Current eight-thread comparison with the default Numba guard (2026-11-17)

The Python FMM proximity guard now uses compiled occupied bins and parallel
per-target queries by default. It preserves the physical/image exclusion
predicate and first-target errors, uses linear storage, and rebuilds the index
for every evaluation. Construction is serial compiled code. Numba is a lazy
FMM dependency; independent direct evaluation still needs just NumPy. Set
`NUMBA_NUM_THREADS=8` as well as `OMP_NUM_THREADS=8` before starting Python.
See [guard behavior and compilation](python.md#parallel-proximity-guard).

The full velocity comparison against the direct Blake kernel in
[RigidMultiblobsWall](https://github.com/annabroms/RigidMultiblobsWall) was rerun
at **eight threads for both methods**, with tolerances `1e-6` and `1e-8`.
Prepared-evaluation times in seconds:

| Ns = Nt | Direct Numba | FMM, tol 1e-6 | Speedup | FMM, tol 1e-8 | Speedup |
|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.0471 | 0.2465 | 0.191x | 0.5343 | 0.0881x |
| 32,768 | 2.594 | 1.310 | 1.98x | 2.835 | 0.915x |
| 131,072 | 40.455 | 5.914 | **6.84x** | 11.560 | **3.50x** |

Speedup is direct Numba time divided by FMM time. The largest `1e-8` FMM time
improves from 16.961 s with the Python guard to 11.560 s, **1.47x faster**, and the speedup against the remeasured direct Numba baseline
increases from 2.40x to **3.50x**. At `1e-6`, FMM first wins in this measured
sweep at 32,768 points; at `1e-8`, the first measured win remains at 131,072.
There is no new one-thread run, so this rerun does not measure an updated
one-to-eight-thread speedup.

The final 131,072-point `1e-8` sample spends 0.0285 s in the guard, 9.662 s in
the Stokes call and 2.064 s in the Laplace call, totaling 11.764 s. The final
`1e-6` sample records 0.0296 s guard, 4.862 s Stokes and 1.012 s Laplace,
totaling 5.914 s. These are individual samples; the table uses median complete
FMM timings. The preceding `1e-8` run's guard took 5.40 s. Native FMM calls now
dominate the evaluation cost.

#### Timing scope, accuracy and provenance

Geometry, signed random forces and seeds match the earlier balanced large
sweep: lateral coordinates in the unit square, source heights `[0.2,0.4]`,
target heights `[0.1,1.1]`, wall zero, viscosity one, point forces, no periodicity
or self-term. Near-wall correction and constituent diagnostics are disabled.
All target velocities are computed. Reference checks select 64 targets and sum
**all sources** independently; full-solve speedup is not extrapolated.

Each direct Numba entry is one timed full solve after signature/runtime warmup,
with no variability estimate, reused across the two tolerances. FMM entries
are medians of three timed full solves after an exact-case warmup. At the
largest size, FMM samples range from 5.840 to 5.922 s at `1e-6` and 11.483 to
11.764 s at `1e-8`. Prepared timings exclude geometry preparation and the
Numba adapter, but include normalization, guard construction/queries, packing,
both native calls and assembly. Imports and JIT compilation are excluded.
Paired end-to-end times at 131,072 points are 40.688 s direct Numba, 6.115 s
FMM at `1e-6` and 11.760 s FMM at `1e-8`, giving 6.65x and 3.46x speedups.

Sampled relative errors at the largest size are 4.46e-15 for direct Numba,
3.63e-6 for FMM at `1e-6`, and 5.00e-9 for FMM at `1e-8`. Full-field
FMM-versus-Numba relative disagreement is 3.17e-6 and 3.66e-9 respectively.
Requested FMM tolerance applies to native kernels and does not guarantee
relative accuracy of the assembled half-space velocity. All dedicated and
benchmark accuracy checks pass, preserving raw wall residuals.

All **22 Python acceptance tests** pass with the user-installed dependencies at
eight threads; no tests are missing. The integrated guard also passes 32
proximity fixtures and four invalid-span checks at each of one/eight threads.
These cover inclusive boundaries, images, clusters, collisions, empty arrays
and scales from `1e-200` to `1e200`. The full MATLAB regression suite passes
at one thread as required by the repository instructions.

The AMD Ryzen 7 PRO 8840HS, Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3,
Numba 0.65.1 and installed FMM3D OpenMP wheel are unchanged. Numba reports eight
threads; both loaded native extensions report `omp_get_max_threads()==8`.
BLAS is limited to one thread and dynamic OpenMP teams are disabled. Binary
fingerprints and build flags are recorded in the earlier provenance section
below; new JSON also fingerprints the production Numba guard source.

From the sibling RigidMultiblobsWall checkout:

```bash
experiments/.venv-blakeComparison/bin/python experiments/blakeComparison.py \
    --sizes 4096 32768 131072 --threads 8 --tolerances 1e-6 1e-8 \
    --repeats 3 --direct-repeats 1 --balanced-only --paired-timings \
    --require-openmp --output experiments/results/blakeComparison-numba-guard
```

See the sibling experiment's
[full report](../../RigidMultiblobsWall/experiments/blakeComparison-results.md),
[updated plots](../../RigidMultiblobsWall/experiments/results/blakeComparison-numba-guard/blakeComparison.png),
[CSV](../../RigidMultiblobsWall/experiments/results/blakeComparison-numba-guard/blakeComparison.csv),
and [raw JSON](../../RigidMultiblobsWall/experiments/results/blakeComparison-numba-guard/blakeComparison.json).
Generated outputs remain in ignored experiment directories and can be recreated
with the command above. Earlier measurements below retain the serial Python
guard and are historical.

### Earlier OpenMP comparison with the serial Python guard (2026-11-17)

The source count must be sufficiently large, with enough target work, for FMM
to outperform the compiled direct Blake kernel in
[RigidMultiblobsWall](https://github.com/annabroms/RigidMultiblobsWall).
The rerun uses the same 4,096 / 32,768 / 131,072 balanced sizes, seed 927 and
requested tolerance `1e-8` as the earlier large-size Python comparison.
It measures complete wall Stokeslet velocities, not free-space Stokeslets alone.

Prepared-evaluation times in seconds:

| Ns = Nt | Numba, 1 thread | FMM, 1 thread | Numba, 8 threads | FMM, 8 threads |
|---:|---:|---:|---:|---:|
| 4,096 | 0.156 | 1.537 | 0.0494 | 0.700 |
| 32,768 | 10.046 | 8.907 | 2.686 | 3.982 |
| 131,072 | 160.788 | 38.877 | 40.725 | 16.961 |

FMM first wins in the measured one-thread sweep at 32,768 points, and in the
matched eight-thread sweep at 131,072 points. At the largest size, it is
**4.14x faster than one-thread Numba** and **2.40x faster than eight-thread
Numba**. Its own one-to-eight-thread speedup is **2.29x**. These are measured
crossover observations, not universal thresholds.

The machine is an AMD Ryzen 7 PRO 8840HS (8 cores / 16 logical CPUs), running
Python 3.10.12, NumPy 2.2.6, SciPy 1.15.3 and Numba 0.65.1. Separate fresh
processes set `OMP_NUM_THREADS=NUMBA_NUM_THREADS` to 1 or 8, disable dynamic
OpenMP teams and hold BLAS threads at one. The exact Numba signature/runtime is
warmed by compilation and correctness checks. Each Numba entry is **one timed
full solve**, so it provides no timing-variability estimate. Each FMM entry is
the median of **three timed full solves**, after an additional exact-case
warmup. No speedup is extrapolated from a subset.

### Workload, timing scopes and achieved accuracy

Sources and targets have independent lateral coordinates in a unit square;
source heights are `[0.2,0.4]`, target heights `[0.1,1.1]`, the wall is at zero,
and viscosity is one. Forces are signed normal random vectors. The Numba source
and target radii are zero, periodic lengths are zero, and there are no
coincident source/target points or added finite-radius self-mobilities.
Optional near-wall correction and constituent diagnostics are disabled during
timing. Both evaluators receive the same translated/recentered geometry.

Paired timing records setup and application within each solve. Prepared time
excludes common geometry preparation and Numba input conversion; FMM's
proximity guard, strength packing, both native calls and assembly remain
included. End-to-end time adds common geometry checks and evaluator preparation.
At 131,072 points, end-to-end times are 160.999 s versus 39.080 s at one thread,
and 40.929 s versus 17.156 s at eight threads, for Numba versus FMM respectively.

The serial Python guard limits application-level scaling. In the last
131,072-point eight-thread sample, it takes 5.40 s of the 16.59 s prepared
invocation; the native Stokes and Laplace calls take 9.20 s and 1.98 s.
This is an individual sample breakdown, not a sum of median component times.
FMM trees and expansions are rebuilt on every application.

All dedicated correctness fixtures pass at both thread settings, including
independent direct Blake fields, shallow sources, exact-wall/near-wall targets,
reciprocity, viscosity scaling and linearity. Each large-case accuracy check
uses 64 selected targets and **every source** in the independent direct sum.
At 131,072 points, sampled global relative errors are **4.46e-15 for Numba**
and **5.00e-9 for FMM**, at requested tolerance `1e-8`. Full-field FMM-versus-
Numba relative disagreement over all targets is **3.66e-9**. Both thread settings
produce these error metrics. All **19 Python acceptance tests** also pass with
the newly installed user-wide bindings and `OMP_NUM_THREADS=8`; no FMM tests
are missing. Raw wall residuals are preserved, and no benchmark assertion
requires a speedup.

### OpenMP build and provenance

The previously imported Python extensions had no OpenMP linkage. The rerun
uses FMM3D `2.1.0`, built from source revision
`0e13b3deb5e2cb35f9fe691464c00e2700208ad5` with GNU Fortran 11.4.0,
Meson optimization level 3, `-fopenmp` on Fortran compilation and both C/Fortran
link steps. Both installed Stokes/Laplace extensions link `libgomp` and expose
OpenMP symbols. Runtime queries through each extension confirm the requested
maximum thread counts of one and eight. Timings above measure the resulting
speedup; a runtime limit alone does not establish how many threads every phase
uses. See [the build commands](dependencies.md#python-bindings-with-openmp).

The verified wheel was installed into both the experiment environment and the
user Python environment; source and MATLAB installations were left unchanged.
The wheel SHA-256 is:

```text
65a4cf5b7563631685cb702599c00d3faabe29ddff25699eda2cbc046bfd7449
```

Loaded extension SHA-256 fingerprints:

| Extension | SHA-256 |
|---|---|
| `lfmm3d_fortran` | `75fa7397d3fa0d7385e0b844ab4fc3438b593693d2515b79059aa00426018a3f` |
| `stfmm3d_fortran` | `d3dba90abb86b9ed1f78b750e0efa40dc83c2b5bfa0fa7a0d6f79321bee413d9` |

### Reproduce the comparison

From the sibling RigidMultiblobsWall checkout containing the experiment driver:

```bash
experiments/.venv-blakeComparison/bin/python experiments/blakeComparison.py \
    --sizes 4096 32768 131072 --threads 1 8 --tolerances 1e-8 \
    --repeats 3 --direct-repeats 1 --balanced-only --paired-timings \
    --require-openmp --output experiments/results/blakeComparison-openmp
```

Use an interpreter with NumPy, SciPy, Numba, Matplotlib and the threaded FMM3D
bindings. `--require-openmp` fails before benchmarking if both imported FMM
extensions do not expose the worker's requested OpenMP runtime limit. Output
includes PNG/PDF scaling and accuracy plots, CSV measurements, and a JSON
record with all repeated samples, geometry seeds, package/source/binary
fingerprints, thread settings and checks. Original serial-build reports remain
separate. The old 131,072-point observation (approximately 39 s for eight-thread
Numba versus 40 s for serial FMM) describes the former build and is superseded
by the verified OpenMP measurements above.
