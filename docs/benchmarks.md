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

The number of sources must be sufficiently large, with enough target evaluations,
for the FMM to outperform the compiled direct Blake velocity kernel in
[RigidMultiblobsWall](https://github.com/annabroms/RigidMultiblobsWall).
For small and moderate problems, its Numba kernel (with zero source and target
radii for point forces) can be faster because direct summation avoids FMM tree
construction and the overhead of the two FMM calls.

In a measured Python comparison with equal source/target counts and FMM tolerance
`1e-8`, FMM first overtook one-thread Numba at 32,768 sources and targets. At
131,072 of each, FMM took 40.5 s versus 158.3 s for one-thread Numba; eight-thread
Numba and FMM were near parity at 39.0 s and 40.4 s. The loaded FMM3D binaries
appeared serial. Numba times were single full-solve measurements; FMM times were
medians of three full solves after warmup. The crossover depends on source and
target counts, geometry, accuracy, hardware, threading and the FMM build; these
measured counts are not a universal threshold.
