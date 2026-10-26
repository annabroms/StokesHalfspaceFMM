# Verification record

Executed on 2026-10-26 with MATLAB R2025a and the external FMM3D installation
identified in [dependencies.md](dependencies.md). Generated artifacts are local
under `results/` and intentionally ignored by Git.

## Regression suite and independent diagnostics

The full regression suite contains 16 tests, including computational function
self-tests. It checks independent velocity formulations, free-space normalization,
charge/dipole signs, 1200-source FMM fixtures, singleton/empty inputs, errors,
duplicate-source addition, prepared geometry, block sizes, physical translations,
viscosity/linearity, reciprocity, free-space convergence, incompressibility and
analytic derivative step studies. Pressure, gradient ordering and full traction
are compared against the independently differentiated Blake tensor.

Final verification in the installed repository: **16 passed, 0 failed, 0
incomplete**. MATLAB Code Analyzer reported **zero findings** across all 25
MATLAB files. The complete output is in `results/final_verification.log`.

The readable validation driver passed 96 orientation/source-count/height/method
cases and 21 dimensional-scaling cases. Heights range from 1e-6 to 1e3; the
nonzero wall height is 0.6. Forces are each Cartesian direction and random
orientations; targets are on, near and far from the wall. Dimensional scaling
uses seven factors from 1e-9 to 1e9 while keeping forces and viscosity fixed.

| Method | Maximum constituent-normalized wall residual | Maximum dimensional-scaling error, normalized |
|---|---:|---:|
| Direct Blake | 2.11e-16 | 4.66e-16 |
| Direct GGV | 1.89e-16 | 4.60e-16 |
| FMM GGV | 3.11e-16 | 7.48e-16 |

These small fixtures mainly verify conventions and roundoff-level cancellation;
they are not evidence of multipole accuracy at arbitrary sizes. The 1200-source
regression fixtures and the benchmark sweep provide separate FMM checks.

Outputs: `validation.csv`, `dimensional_scaling.csv`, `validation.mat`, and
`validation.png`/`.pdf`. Exact zero errors are displayed at 1e-18 **only in the
plots**; raw data retain zero. No velocity is projected onto the wall condition.

## Size, tolerance and block benchmarks

The executed quick profile has 45 cases: Ns=32,128,512,2048,8192; Nt/Ns=1/4,1,4;
source-layer base heights 1e-5,0.01,1. Total targets include a separate wall subset
and reach 32768. All timings use five repetitions, medians and warmed interfaces.
Separate processes record OMP_NUM_THREADS=1 and 8 in their MAT metadata.

Above 2e7 pairs, direct comparison uses up to 256 stratified bulk targets and
256 wall targets, summing all sources. The 8192-by-32768 cases therefore validate
512 targets without forming the full interaction matrix. No full speedup is
inferred from these subset timings.

One-thread benchmark results at requested tolerance 1e-9:

- Maximum constituent-normalized error: 5.23e-11.
- Maximum constituent-normalized wall residual: 1.88e-11.
- Maximum global relative error: 2.16e-6, in a strongly cancelling thin source layer.
- First measured crossover in several complete comparisons: Ns=2048.
- Largest measured complete-evaluation speedup: 2.06. Other cases remained slower
  than direct evaluation; missing crossover entries are NaN, not extrapolations.

Both 45-case profiles completed. Their artifacts are identified as follows:

| FMM thread setting | Artifact stem | Maximum measured complete speedup |
|---|---|---:|
| OMP_NUM_THREADS=1 | `benchmark_quick_20261026_152645` | 2.06 |
| OMP_NUM_THREADS=8 | `benchmark_quick_20261026_153138` | 2.00 |

The measured errors agree across these thread settings. More threads did not
improve every case: at 8192 sources and 32768 targets, measured total times
were approximately 1.6–2.3 seconds for both profiles, depending on source height.
The raw tables preserve the differences rather than assuming parallel speedup.

The fixed 2048-source, 257-target tolerance sweep demonstrates requested versus
measured accuracy without assuming equality:

| Requested tolerance | Global relative error | Constituent-normalized error | Normalized wall residual |
|---:|---:|---:|---:|
| 1e-3 | 1.76e-3 | 9.05e-6 | 2.59e-6 |
| 1e-6 | 3.07e-6 | 1.32e-8 | 2.83e-9 |
| 1e-9 | 2.75e-9 | 1.90e-11 | 3.06e-12 |
| 1e-12 | 5.46e-12 | 3.80e-14 | 4.52e-15 |

The one-thread 1024-by-1024 block study measured median seconds of 0.0711 for
64/64 blocks, 0.0474 for 128/256, 0.0426 for 256/512, 0.0441 for 512/256, and
0.164 for the source-loop/vectorized-target alternative (1024/1). Retain the
planned default `[256 512]`; this is one fixture, not a universal optimum.
All variants agreed within 2.4e-14 maximum absolute velocity difference.

Benchmark filenames contain timestamps and end in `_scaling.csv`,
`_tolerance.csv`, `_blocks.csv`, `_observations.csv`, `.mat`, `.png`, and `.pdf`.
MAT files contain hardware, OS, MATLAB version, binary checksum, source revision,
thread setting, configuration and seed information. Observed slopes include
small-size overhead and are descriptive rather than asymptotic complexity claims.

The longer `default` and `large` profiles are implemented but were not executed
as part of this verification. No runtime or accuracy claims are made for those
unrun sizes.
