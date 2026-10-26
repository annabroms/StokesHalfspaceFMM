# StokesHalfspaceFMM

**This implementation is based on Z. Gimbutas, L. Greengard and S. Veerapaneni,
“Simple and efficient representations for the fundamental solutions of Stokes
flow in a half-space,” Journal of Fluid Mechanics 776 (2015), R1.**
[Published paper](https://doi.org/10.1017/jfm.2015.302) ·
[Open-access preprint](https://arxiv.org/abs/1504.00102v2).

The independent direct reference implements J. R. Blake, “A note on the image
system for a stokeslet in a no-slip boundary,” Proc. Cambridge Philos. Soc. 70
(1971), 303–310, [equation (16)](https://doi.org/10.1017/S0305004100049902).

MATLAB evaluation of Stokes flow above an infinite stationary no-slip wall,
using the paper's half-space image representation with a Papkovich–Neuber
correction. This representation can be evaluated directly or with two calls to
the external FMM3D library. An independently implemented direct Blake tensor
provides a reference. Pressure, target velocity gradients and complete traction
are also available through `blake.fields`.

## Setup

Tested with MATLAB R2025a on Ubuntu 22.04.5 LTS (Linux x86-64); FMM evaluation requires
a compiled FMM3D MATLAB installation. Direct evaluation requires no external
library or additional MATLAB toolbox. Build FMM3D following its own instructions:
<https://fmm3d.readthedocs.io/en/latest/install.html>.

From the repository folder, run:

```matlab
setpath;
```

This adds the repository root, MATLAB package directory, examples, benchmarks
and tests. The script locates the repository from its own file location and
expects FMM3D at `../FMM3D/FMM3D/matlab`. For a different dependency location:

```matlab
addpath('matlab');
manifest = blake.setup('/path/to/FMM3D/matlab');
```

Only the selected dependency directory is added. `blake.setup` checks that both
wrappers and the MEX resolve to one directory. See
[dependencies and the tested system](docs/dependencies.md) for requirements,
installation guidance and reproducibility information.

## Velocity

```matlab
sources = [0 0.5; 0 0.2; 1 0.7];   % 3 x Ns
forces  = [1 -0.3; 0 0.4; 0.2 1];  % point forces applied to the fluid
targets = [0.1 0.4 0.7; 0.2 0.3 0.1; 0 0.5 1.2]; % independent 3 x Nt
opts = struct('method','fmm','mu',1.5,'tol',1e-9);
[u, info] = blake.velocity(sources,forces,targets,opts);
```

`u` is physical velocity, always `3 x Nt`, **not** viscosity-scaled velocity.
SI inputs give m/s. Forces have no quadrature weights added implicitly.
Defaults are `mu=1`, `wall_height=0`, `tol=1e-9`, `method='fmm'`,
`block_size=[256 512]` (target/source block sizes), `diagnostics=false`,
and `stokes_backend='auto'`.
Methods are `blake`, `ggv-direct`, and `fmm`. The option name `ggv-direct`
abbreviates the three authors' surnames and selects direct evaluation of their
half-space image representation; it is not a formal name for the method.
Coordinates and forces must be
finite, real doubles with components in rows and points in columns.

Set `opts.wall_height` for a wall at a nonzero height. Sources must be strictly
above it; targets may lie on it. All three wall velocity components are actually
evaluated: the code never replaces their residual with zero.

Exact physical source/target coincidences throw `blake:SingularInteraction`,
even for zero forces. There is no self-omission convention. Duplicated sources
otherwise contribute additively. Empty sources produce zero fields, and empty
target sets preserve empty output shapes. No regularization is applied.

```matlab
geom = blake.prepare(sources,targets,0);
[u,info] = blake.apply(geom,forces,opts);
```

Prepared geometry contains translated/recentered coordinates, reflected images,
heights, source identities and counts. Treat it as opaque and immutable.
Geometry preparation and its proximity assessment are reused; FMM trees and
expansions are rebuilt on every application. To change the wall, prepare a new geometry. `info` separates strength,
proximity guard, kernel, assembly, diagnostic and total timings.

`[geom,prep_info]=blake.prepare(sources,targets,wall_height,precheck_fmm)` also
returns separate geometry and guard timings. `precheck_fmm` defaults to `true`.
The cached assessment stores pass/fail status, threshold and offending pair;
unsafe geometry raises `blake:FMMResolution` when FMM evaluation is requested,
while direct evaluation remains available. Set the fourth argument to `false`
to defer assessment; FMM application still checks unchecked or older geometry.
One-shot direct evaluations skip this FMM-only assessment. Repeated application
to checked geometry reports `guard_reused=true`, `guard_seconds=0`.

`opts.stokes_backend` selects the free-space Stokes implementation within the
half-space calculation. The default is `auto`, so ordinary `blake.velocity`,
`blake.apply`, and velocity-only `blake.fields` calls use `laplace-velocity`
without specifying an option. Requests containing pressure, gradients or
traction use upstream Stokes to obtain the required fields together:

- `auto`: four-potential `laplace-velocity` for velocity-only requests; upstream
  Stokes for pressure or gradients.
- `upstream`: always use the original `stfmm3d` path.
- `laplace-velocity`: explicitly select the target-only four-potential adapter;
  derivative requests raise `blake:UnsupportedBackend`.

The adapter uses the same installed FMM3D binary, without a new MEX or external
library edits. `info.stokes_backend` records the effective choice. The separate
harmonic correction and all physical conventions are preserved.

## Pressure, gradients and traction

```matlab
opts.requested = {'velocity','pressure','gradient','traction'};
opts.normals = repmat([0;0;1],1,size(targets,2)); % supplied unit normals
[a,info] = blake.fields(sources,forces,targets,opts);
% a.velocity: 3 x Nt, a.pressure: 1 x Nt
% a.grad_u(i,j,m) = du_i/dx_j; a.traction: 3 x Nt
```

Velocity is always returned; derivative prerequisites may also be returned.
Pressure is independent of viscosity for fixed forces. Traction is
`(-p*eye(3) + mu*(grad_u + grad_u.'))*normal`, with the sign determined by the
supplied normal. It includes the entire wall correction. Direct Blake derivative
evaluation is a readable pair-by-pair reference and is slower than blocked velocity.

## Tests and validation

```matlab
addpath('tests');
results = run_tests;               % full acceptance, requires FMM3D, no skipped tests
results = run_tests(false);        % direct tests; missing FMM checks are INCOMPLETE
addpath('examples');
report = validate_velocity;        % deterministic metrics, CSV/MAT and PNG/PDF
```

Each computational function also runs a small deterministic check with no inputs,
for example `blake.internal.direct.blake_pair()` or `blake.velocity()`.
FMM self-tests fail explicitly if the dependency is absent.

Tests cover independent formulas, each force direction, random forces, unequal
counts, duplicates, wall heights, dimensional scaling from 1e-9 to 1e9,
incompressibility and derivative step studies, reciprocity and the free-space
limit. Nontrivial FMM fixtures use 1200 sources, not just direct leaf sums.

With `diagnostics=true`, `info.constituent_scale` sums magnitudes of individual
physical Stokes, reflected Stokes, z*grad(phi) and phi contributions **before**
cancellation. This is an optional **O(Ns*Nt)** direct calculation. For large
problems request it only on a representative target subset, with all sources.
Its time is separately reported and included in total time.

`blake.metrics(u,reference,scale)` reports maximum pointwise absolute error,
global Frobenius relative error (NaN for a negligible reference), and maximum
constituent-normalized absolute error. Wall references are zero; their relative
velocity error is undefined. FMM regression budgets are empirical, not guarantees
that requested tolerance bounds the relative error of the corrected field.

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

It evaluates **free-space Stokeslet velocity only**, using Ns ordinary sources
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

## Limitations

- Horizontal walls and point-force singularities only; no regularized forces,
  periodic boundaries, finite walls or MFS solvers.
- The FMM adapter rejects targets within 64*eps*L of physical **or reflected**
  sources, conservatively above the backend exclusion threshold. This guard is
  domain-relative and scales with the geometry; direct methods remain available.
- Double precision cannot recover separations already lost in the input, and
  extreme geometry or force ranges can overflow. Severe cancellation limits
  final relative accuracy.
- The upstream Stokes MATLAB wrapper hides its internal `ier`. The new velocity
  adapter checks the Laplace `ier` explicitly; pressure/gradient requests still
  have the upstream status limitation. Both paths check output validity.
- No FMM tree/expansion reuse is exposed by this interface.

See [formulas and derivative conventions](docs/formulas.md),
[dependencies and tested system](docs/dependencies.md), and [verification record](docs/validation.md).
