# MATLAB interface

[Back to the main README](../README.md)

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
`stokes_backend='auto'`, and `fmm_backend='two-call'`.
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
half-space calculation:

- `auto` (default): use the updated `stfmm3d` for every requested field.
- `upstream`: explicitly select that same updated Fortran implementation.
- `laplace-velocity`: explicitly select the legacy MATLAB four-potential
  reference adapter for velocity comparisons. It is no longer selected by
  default; derivative requests raise `blake:UnsupportedBackend`.

Ordinary `blake.velocity`, `blake.apply`, and velocity-only `blake.fields`
calls use `stfmm3d(tol,srcinfo,0,targets,1)`. The updated Fortran code performs
the four-potential composition and recentering internally, with Laplace flags
`(0,2)`: no source fields and **no Laplace Hessians**. The separate harmonic
correction also calls `lfmm3d` with flags `(0,2)`. No standalone
`stfmm3d_laplace` wrapper is used or required.

Pressure requests use Stokes flag 2 and harmonic Laplace flag 2, also without
Hessians. Velocity-gradient and traction requests use flag 3 and require
Laplace Hessians to compute those derivatives. `info.stokes_backend` records
the effective choice; the historical name `upstream` denotes the selected
updated installation, not the sibling `FMM3D_basic` checkout.

## Optional single-call Fortran backend

The existing two-call implementation remains the default. To build and select
one folded four-potential Laplace call instead:

```matlab
setpath;
blake.build_onecall;
opts = struct('fmm_backend','one-call','tol',1e-9);
[u,info] = blake.velocity(sources,forces,targets,opts);
```

The new backend supports velocity and pressure, uses no Laplace Hessians,
and leaves the external FMM3D installation unchanged. Gradients/traction remain
available through the default two-call path. Run `compare_fmm_calls` separately
to compare fields, raw wall residuals, reciprocity and scaling of both methods.
See [the derivation, build instructions and measured comparison](one-call-fmm.md).

## Wall rotlets, potential dipoles and stresslets

`blake.rotlet`, `blake.potential_dipole` and `blake.stresslet` provide complete
wall-corrected velocity with blocked direct application. They preserve the
component-first layout, physical units, viscosity and wall-height conventions.
The stresslet uses an oriented double force with prescribed unit source normals;
each family has three linear strength coefficients per source. See
[definitions, derivations, references and self-tests](singularities.md).

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
blake.build_onecall;               % build optional backend for its acceptance tests
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
