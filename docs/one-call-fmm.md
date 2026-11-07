# One FMM call versus two for half-space Stokes flow

The existing two-call implementation remains the default. The optional
`fmm_backend='one-call'` performs the same Gimbutas--Greengard--Veerapaneni
half-space representation in one compiled Fortran Laplace FMM invocation.
The external FMM3D installation is linked, not modified. No MFS dependency
is introduced.

| | Existing `two-call` | Optional `one-call` |
|---|---|---|
| Composition | Updated Stokes FMM plus scalar Laplace charge/dipole wall correction | One Laplace FMM for four folded charge/dipole potentials |
| Sources per call | Physical + images for Stokes; images only for the correction | Physical + images for all four densities |
| Dipole work | Confined to the scalar wall correction | Enabled for four densities; zero entries are not skipped by the general kernels |
| Trees | Two trees per evaluation | One tree per evaluation |
| Implementation | Existing Stokes Fortran plus MATLAB wall assembly | Local Fortran packing and complete-field reconstruction |
| Recentering | Shared MATLAB wall shift/horizontal center, then updated Stokes routine recenters its potentials | Same MATLAB preparation, then a common center for the folded potentials and the corresponding wall-height correction |
| Timing | MATLAB `tic`/`toc` around backend interfaces and complete/prepared calls | Same MATLAB timing; MEX boundary includes native packing and reconstruction |
| Outputs | Velocity, pressure, gradient, traction | Velocity and pressure |
| Laplace derivatives for velocity/pressure | Potentials and gradients | Potentials and gradients |
| Laplace Hessians for velocity/pressure | None with updated FMM3D | None; enforced by the native gateway |

Gradient/traction requests with `one-call` raise `blake:UnsupportedBackend`.
The existing default path retains these outputs, including their necessary
Hessians. Neither implementation forces a computed wall velocity to zero.

## Representation and recentering

The representation is from Z. Gimbutas, L. Greengard and S. Veerapaneni,
*Simple and efficient representations for the fundamental solutions of Stokes
flow in a half-space*, JFM 776 (2015), R1. See [the existing derivation](formulas.md)
for normalization, reflection and source-dipole signs.

[Geometry preparation](../matlab/+blake/prepare.m) puts the wall at z=0 and
recenters the horizontal coordinates. [Strength preparation](../matlab/+blake/+internal/+halfspace/strengths.m)
supplies the physical/reflected Stokes strengths and the harmonic charge q and
dipole d on each image. These helpers and the existing proximity guard are reused.

For K=1/(4*pi*r), the ordinary four Stokes potentials are

```
psi_j = sum (sigma_j/2)*K, j=1,2,3
psi_4 = sum ((Y dot sigma)/2)*K
```

Here Y=y-c and X=x-c use one center for physical sources, images and targets.
The harmonic wall potential is phi=sum(q*K + d.grad_y(K)), and the target
height above the wall is z=X_3+c_3. The complete field can therefore be written

```
Phi_1 = psi_1
Phi_2 = psi_2
Phi_3 = psi_3 + phi
Phi_4 = psi_4 - c_3*phi

mu*u_i = Phi_i - sum_j X_j*d_i(Phi_j) + d_i(Phi_4)
p      = -2*sum_j d_j(Phi_j)
```

The `-c_3*phi` term must be included when the vertical center is nonzero.
It preserves the original wall height in the PN correction. Pressure is not
divided by viscosity. The factors 1/2 reproduce the Stokeslet's 1/(8*pi)
normalization from the Laplace kernel's 1/(4*pi).

The new [Fortran routine](../native/blake_onecall.f90) packs these four potentials
and calls `lfmm3d` once with source flag 0 and target flag 2. Small unused
Hessian arguments satisfy the Fortran interface; no Hessian computation is
requested. The [MEX gateway](../native/blake_onecall_mex.c) intercepts actual
`lfmm3d` invocations at link time, counts them and rejects any call other than
one four-density invocation with flags `(0,2)`.

## Build and use

The initial build script supports Linux with GNU C/Fortran and OpenMP. Select
a compiled FMM3D installation containing `lib-static/libfmm3d.a`, then run:

```matlab
setpath;
build = blake.build_onecall;
opts = struct('method','fmm','fmm_backend','one-call','tol',1e-9);
[u,info] = blake.velocity(sources,forces,targets,opts);
opts.requested = {'pressure'};
[a,info] = blake.fields(sources,forces,targets,opts);
```

`info.fmm_call_count` comes from the native call counter. The native Laplace
status is checked explicitly. `info.laplace_source_flag` and
`info.laplace_target_flag` report the observed flags. All timings use MATLAB
`tic`/`toc`; the local Fortran wrapper contains no timers. Backend timings include the complete MATLAB
interface call, including native packing and reconstruction. Complete public-call
timing also includes geometry, checks and physical scaling. Prepared geometry
reuses checks, not FMM trees.

The build stores the static-library, source and MEX checksums and compiler
settings in ignored `native/build/build.json`. Rebuild the local backend after
changing the selected FMM3D installation, its library or the native sources.
The old `stokes_backend` option continues to select the free-space method for
`two-call`; the legacy MATLAB adapter is not part of the new backend.

## Standalone comparison and correctness tests

From the repository folder:

```matlab
setpath;
blake.build_onecall;
report = compare_fmm_calls;
% Smaller exploratory run, or save the raw repetitions and tables:
% report = compare_fmm_calls(struct('sizes',[1024 4096], ...
%     'ratios',[1],'repetitions',3,'save',true));
```

[compare_fmm_calls](../benchmarks/compare_fmm_calls.m) is separate from the
correctness suite. It reports:

1. Velocity and pressure errors against the independent direct Blake reference,
   plus implementation differences. Fixtures include 1200 sources, shallow
   layers, nonzero wall height/viscosity, large horizontal translations, and
   a high target that exercises vertical recentering.
2. Raw no-slip residuals in all three wall velocity components. Absolute and
   constituent-normalized errors are reported; relative error against zero
   wall flow is undefined.
3. Reciprocity of a full Cartesian pair tensor and of distributed force work
   after exchanging two disjoint clouds. Norm-product normalization avoids
   division by a small signed work sum.
4. End-to-end and prepared-geometry **velocity** timings as Ns and Nt grow.
   Pressure is checked for accuracy above but is not requested in these timings. Each method is
   warmed, execution order alternates and every repetition is retained.
   Summaries show median timings, speedup and the range of paired speed ratios.
   Direct checks on stratified wall/bulk targets sum all sources outside timers.

Both implementations use recentering throughout this comparison; this is a
comparison of where the Laplace composition is performed, not a switch between
centered and uncentered evaluation. The tall-target and translated fixtures
check that their different recentering steps preserve the physical fields.

The default comparison uses Ns=1024,4096,16384,65536, Nt/Ns=1/4,1,4,
five repetitions, tolerance 1e-8 and fixed seed 481516. Ten percent of timing
case targets are on the wall; source heights are in [0.1,1.1]. Run separate
fresh MATLAB processes with `OMP_NUM_THREADS=1` or another desired thread count.
`save=true` writes a uniquely named MAT report and CSV tables under ignored
`results/`; provenance includes both the selected FMM3D binary and local build.

The new [acceptance tests](../tests/test_onecall.m) join the existing suite:

```matlab
results = run_tests;
```

Build the optional backend before running the full acceptance suite. Missing
native dependencies are incomplete checks, not passes. The backend also has
an independent no-argument self-test: `blake.internal.fmm.onecall`.

## Performance tradeoff

One call shares the tree and translations, but enables charge/dipole work for
all four densities on the combined physical/image source list. The dipoles are
needed for the wall correction; only the image sources carry nonzero dipoles.
The old Stokes call is charge-only, while its separate harmonic call has only
image sources. FMM3D selects charge-and-dipole evaluation routines for the
combined call, so zero dipole entries still incur overhead. With many targets,
this extra target work can outweigh the saved tree construction. The combined
source list can also change the tree subdivision and near-field work.
Consequently one call need not be faster. Speedup is defined as two-call time
/ one-call time; values below one favor the existing implementation. No test
asserts that the new implementation is faster.

## Possible speed improvements

The first small experiment would be **leaf-size tuning** for this four-density
charge/dipole workload. FMM3D's default subdivision selector is not passed the
number of densities. Its existing `lfmm3d_ndiv` entry point allows control of
leaf occupancy and subdivision criteria. Changing those choices trades direct
near-field work against tree/expansion work; it needs measurements over several
source/target ratios and the same accuracy checks. All comparisons should still
be timed from MATLAB. This tuning has not been implemented or measured here.

A larger optimization would let the kernels **skip zero dipoles by source or
density**. Physical sources have no wall-correction dipoles, and the first two
folded potentials have none either. The public FMM dipole flag is global, so
simply turning it off would remove the required image correction. Skipping the
zero work while sharing one tree would require changes below the current
wrapper interface.

For repeated evaluations, `blake.prepare`/`blake.apply` already avoid repeated
geometry checks. Reusing packed coordinates or parallelizing native packing
and reconstruction could reduce wrapper costs, but would leave the extra
charge/dipole kernel work in place. None of these possibilities establishes a
speedup without a MATLAB timing comparison; the backend is unchanged by the
comparison's readability refactor.

## Validation measurements (2026-11-07)

- Full acceptance suite: **28 passed, 0 failed, 0 incomplete**,
  10.424 s. Code Analyzer: zero findings in seven changed/new MATLAB files.
- MATLAB 25.1.0.2943329 (R2025a); AMD Ryzen 7 PRO 8840HS w/ Radeon 780M Graphics; `OMP_NUM_THREADS=1`.
- Scaling run: default sizes/ratios above, five repetitions, tolerance 1e-8,
  seed 481516. Complete velocity calls and prepared calls are measured separately.
  Every reported duration is measured with MATLAB `tic`/`toc`, outside the native call.
- The native counter verified one call with flags (0,2); full correctness
  tests also cover pressure, vertical recentering, wall height, viscosity,
  empty data, coincidences and explicit rejection of gradients.

Across the four field fixtures, the largest one-call velocity relative error
against independent direct Blake evaluation was 8.670334e-10;
its largest constituent-normalized error was 2.612786e-11.
The largest pressure relative error was 4.011088e-10.
The largest one-call/two-call velocity difference, normalized by the common
constituent scale, was 2.825266e-11.

Flow at the wall was zero to numerical accuracy, not overwritten: the largest
raw one-call wall residual was 3.374325e-09,
and the largest constituent-normalized wall residual was
1.225643e-11.

With 4096 and 2048 sources in the exchanged clouds,
the one-call norm-product-normalized work defect was 4.671024e-12;
the Cartesian pair-tensor defect was 5.409543e-16.

Selected complete-call medians:

| Sources | Targets | Two calls (s) | One call (s) | Speedup |
|---:|---:|---:|---:|---:|
| 1,024 | 1,024 | 0.038666 | 0.033259 | 1.163 |
| 4,096 | 4,096 | 0.337992 | 0.283554 | 1.192 |
| 16,384 | 16,384 | 0.640452 | 0.671824 | 0.953 |
| 16,384 | 65,536 | 1.294753 | 1.467628 | 0.882 |
| 65,536 | 65,536 | 4.168853 | 3.944738 | 1.057 |
| 65,536 | 262,144 | 6.396779 | 6.529610 | 0.980 |

There is **no uniform speedup that grows with problem size**. Some balanced
cases improved, while some higher-target-count cases were slower. In the
65,536/65,536 case, paired complete-call speed ratios ranged from
1.015 to 1.105.
Across all measured configurations, prepared-geometry speedups ranged from
0.868 to 1.379.
The existing two-call implementation remains the default.

Use total and prepared timing columns to compare performance. The `*_fmm_s`
columns time the backend interfaces from MATLAB for both implementations. The
new backend includes packing and reconstruction inside its MEX boundary; the
old backend performs its final half-space assembly outside its FMM interfaces.
Complete-call timings include all costs for both implementations.

Full scaling data: [MAT report](../results/tp26da5141_b92e_4aa5_afd8_a8d2fe48d282_fmm_calls.mat),
[all timing summaries](../results/tp26da5141_b92e_4aa5_afd8_a8d2fe48d282_fmm_calls_timings.csv),
[raw repetitions](../results/tp26da5141_b92e_4aa5_afd8_a8d2fe48d282_fmm_calls_samples.csv),
[field accuracy](../results/tp26da5141_b92e_4aa5_afd8_a8d2fe48d282_fmm_calls_accuracy.csv),
[reciprocity](../results/tp26da5141_b92e_4aa5_afd8_a8d2fe48d282_fmm_calls_reciprocity.csv).
Generated reports are ignored by Git. Re-run the driver with `save=true` to
create reports on another installation.

Dependency provenance from this scaling run:

- Selected Stokes wrapper: `FMM3D/matlab/stfmm3d.m`.
- Selected FMM3D MEX SHA-256: `ac1190b3485d4ad2fa2df77ffd2ec5d659aa66f8a7d2ec62f2c1fe239e1dc650`.
- Linked static library SHA-256: `d12369e678370951e7d260f597ede596cb6fbb3e60b997470181fa1034a0ba43`.
- One-call MEX SHA-256: `8b1a0526e4a9e366c466117d13a9de059762ae183370f947dd6a8c4fd328bc40`.
- The complete build record, compiler settings and source checksums are in
  `native/build/build.json` and in the saved report metadata.
