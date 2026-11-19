# StokesHalfspaceFMM

MATLAB and Python evaluation of Stokes flow above an infinite stationary
no-slip wall, using the image representation of **Z. Gimbutas, L. Greengard
and S. Veerapaneni**, *Simple and efficient representations for the fundamental
solutions of Stokes flow in a half-space*, JFM 776 (2015), R1.
[Published paper](https://doi.org/10.1017/jfm.2015.302) ·
[Open-access preprint](https://arxiv.org/abs/1504.00102v2).

An independent direct reference implements J. R. Blake's original tensor,
[equation (16), 1971](https://doi.org/10.1017/S0305004100049902).
Both interfaces use component-first `3 x N` arrays and return physical velocity.

## What is supported, compared with STKFMM

All rows are quantities above a no-slip wall. "Direct" means blocked O(Ns*Nt)
summation. The STKFMM column covers the wall class `StkWallFMM` in
[STKFMM](https://github.com/annabroms/STKFMM/tree/local_python), as on the
`local_python` branch of this fork at commit `1cea52a`.

| Quantity above the wall | MATLAB FMM | MATLAB direct | Python FMM | Python direct | STKFMM `StkWallFMM` |
|---|:-:|:-:|:-:|:-:|:-:|
| Stokeslet (point-force) velocity | Yes | Yes | Yes | Yes | Yes (`Stokes`) |
| Pressure | Yes¹ | Yes | Yes | Yes | — |
| Velocity gradient and traction | Yes² | Yes | Yes | Yes | — |
| Rotlet, potential-dipole and oriented-stresslet velocity | — | Yes | — | — | — |
| RPY velocity for finite-radius blobs, plus its Laplacian | — | — | — | — | Yes (`RPY`) |
| Periodic in x or in x and y | — | — | — | — | Yes³ |

¹ Both the default two-call backend and `fmm_backend='one-call'`.
² Two-call backend only.
³ STKFMM lists `PX` and `PXY` support for both wall kernels. These modes need
STKFMM's precomputed periodic data and were not tested here.

STKFMM's free-space class `Stk3DFMM` has more kernels, such as pressure,
velocity gradients, traction and double layers. These have **no wall
correction** and are not equivalents of the rows above.

Other differences:

- **Only shared quantity.** Point-force velocity is the only quantity both
  libraries compute, so the
  [timing comparison](#fmm3d-versus-stkfmm-with-a-wall) measures only that.
- **Coincident points.** This package raises an error when a source and a
  target coincide.
- **Viscosity.** STKFMM sets viscosity to one, so its output must be divided by
  `mu`. This package takes `mu` as an input and returns physical velocity.
- **Wall position.** STKFMM places the wall at the lower face of its cubic box,
  and all points must lie in the lower half of that box. This package takes a
  `wall_height` option and builds trees from the points themselves.
- **Parallelism and reuse.** STKFMM can run across several processes with
  MPI, and can reuse a tree for new strengths. This package uses OpenMP in a
  single process through FMM3D, and rebuilds its trees on every call.

## Shared FMM3D dependency

Both language wrappers use the same underlying **FMM3D Fortran implementation**.
The default wall velocity requires two calls: Stokeslets at the physical and
reflected sources, followed by a Laplace charge/dipole Papkovich–Neuber correction.

Build FMM3D and the language interfaces you need using its
[installation instructions](https://fmm3d.readthedocs.io/en/latest/install.html).
Rebuild each interface after updating the shared Fortran implementation.
See [dependency selection and backend details](docs/dependencies.md).

## Matlab setup

From the repository root:

```matlab
setpath;
```

Setup finds a sibling FMM3D installation. To select another, set
`FMM3D_MATLAB` before calling `setpath`, or pass its MATLAB directory directly
to `blake.setup`. Direct evaluation requires no FMM3D or additional toolbox.

```matlab
sources = [0 0.5; 0 0.2; 1 0.7];             % 3 x Ns
forces  = [1 -0.3; 0 0.4; 0.2 1];            % 3 x Ns
targets = [0.1 0.4 0.7; 0.2 0.3 0.1; 0 0.5 1.2]; % 3 x Nt
opts = struct('method','fmm','mu',1.5,'tol',1e-9);
[u,info] = blake.velocity(sources,forces,targets,opts);
```

## Python setup

Use Python 3.10 or newer with NumPy, Numba and the compiled `fmm3dpy` interface.
Direct evaluation requires only NumPy. Select the interpreter containing your
FMM3D installation and start Python from the repository root with:

```bash
PYTHONPATH=python python3
```

```python
import numpy as np
import blake

sources = np.array([[0, .5], [0, .2], [1, .7]])
forces = np.array([[1, -.3], [0, .4], [.2, 1]])
targets = np.array([[.1, .4, .7], [.2, .3, .1], [0, .5, 1.2]])
u, info = blake.velocity(sources, forces, targets,
                         mu=1.5, tol=1e-9, return_info=True)
```

See [Python usage and options](docs/python.md) for environment checks and
prepared geometry.

## Evaluation methods

| Choice | MATLAB | Python |
|---|---|---|
| FMM | `method='fmm'` (default) | `method="fmm"` (default) |
| Direct Blake tensor | `method='blake'` | `method="blake"` |
| Direct Gimbutas–Greengard–Veerapaneni image representation | `method='ggv-direct'` | — |

Both interfaces support `correct_near_wall` (off by default) to replace poorly
conditioned close targets with direct Blake sums. See [cutoff choice, accuracy and cost](docs/near-wall-correction.md).

MATLAB offers two FMM compositions: the default
`fmm_backend='two-call'`, and an optional `fmm_backend='one-call'` that folds
the representation into four Laplace potentials. Build the latter with
`blake.build_onecall`; it supports velocity and pressure.
[Derivation, build and comparison](docs/one-call-fmm.md).

Within the MATLAB two-call path, `stokes_backend='auto'` or `'upstream'`
selects the updated FMM3D Stokes implementation. The optional
`'laplace-velocity'` adapter evaluates free-space Stokes velocity through four
Laplace potentials. [Options and output conventions](docs/matlab.md).

## Kernels and fields

Both languages evaluate wall Stokeslet velocity, pressure, velocity gradients
and complete traction through `blake.fields`. These are available by FMM or
by direct summation, as in this Python example:

```python
out = blake.fields(sources, forces, targets, requested=("pressure", "traction"),
                   normals=normals, mu=1.5)
# out["velocity"]: (3, Nt); out["pressure"]: (Nt,), not divided by mu
# out["grad_u"][i, j, m] = du_i/dx_j; out["traction"]: (3, Nt)
```

Gradient and traction requests take the same two FMM calls as velocity, with
higher target output flags. MATLAB returns pressure as `1 x Nt`. MATLAB
additionally provides wall **rotlets**, **potential dipoles** and
**stresslets** through `blake.rotlet`, `blake.potential_dipole` and
`blake.stresslet`, using blocked direct evaluation.

See [MATLAB fields](docs/matlab.md#pressure-gradients-and-traction),
[Python fields](docs/python.md#pressure-velocity-gradients-and-traction) and
[other kernel definitions and tests](docs/singularities.md).
Published no-slip wall derivations for these other singularities are listed in
the [singularity references](docs/singularities.md#checked-references), including
Blake & Chwang (1974), Gimbutas, Greengard & Veerapaneni (2015), and
Spagnolie & Lauga (2012), with equation pointers and normalization notes.

## Numerical conventions and limits

- Sources and forces are `3 x Ns`; targets and velocity are `3 x Nt`.
  Forces act on the fluid, with no implicit quadrature weights.
- Defaults are `mu=1`, `wall_height=0` and `tol=1e-9`.
  Velocity includes division by viscosity; SI inputs give m/s.
- Sources must be strictly above the wall; targets may lie on it. All three
  wall residual components are computed without overwriting them.
- Exact physical source/target coincidences raise errors, even for zero forces.
  Duplicate sources add; empty inputs preserve output shapes.
- The FMM guard rejects unresolved physical or reflected pairs within
  `64*eps*L`, where `L` is the geometry span. Direct evaluation remains
  available for noncoincident pairs.
- Prepared geometry reuses coordinates, but FMM trees and expansions are rebuilt.
  Strong cancellation limits relative accuracy; extreme input ranges can overflow.
  Walls are horizontal, with no periodicity, regularization or MFS dependency.
- Optional constituent diagnostics cost O(Ns*Nt) direct work in bounded blocks.
  Wall checks use absolute or constituent-normalized error; relative error
  against zero wall velocity is undefined.

## Tests and performance

From the repository root, using the installed compiled interfaces:

```bash
OMP_NUM_THREADS=1 matlab -batch "addpath('matlab','tests'); blake.setup; run_tests;"
PYTHONPATH=python OMP_NUM_THREADS=1 python3 python/tests/run_tests.py
```

Build the optional MATLAB one-call backend before full acceptance.
Missing FMM checks are **incomplete**, not passes. Direct-only development uses
`run_tests(false)` in MATLAB or `--direct-only` in the Python test runner.
Computational functions also have independently runnable no-argument self-tests.

Tests compare independent formulas, raw wall residuals, reciprocity, Cartesian
and random forces, unequal counts and dimensional scaling. See the
[MATLAB validation record](docs/validation.md) and
[Python accuracy report](docs/python-validation.md).

Benchmarks are separate from correctness tests and run only when requested.
Small correctness cases do not establish large-system performance.
[Benchmark profiles, timing definitions and the direct Numba comparison](docs/benchmarks.md)
describe the measured tradeoffs and how to reproduce a performance study.

### OpenMP FMM versus direct Numba

The optimized point-force Blake kernel in
[RigidMultiblobsWall](https://github.com/annabroms/RigidMultiblobsWall) substantially
reduces the direct cost. The rerun retains the earlier geometry distribution on an
AMD Ryzen 7 PRO 8840HS, with verified **eight threads for both evaluators**.
Prepared-evaluation times in seconds include the FMM proximity guard, packing,
both native calls and assembly; correction and diagnostics are off:

| Direct kernel / measurement date | Ns = Nt | Direct Numba | FMM, tol 1e-6 | Speedup | FMM, tol 1e-8 | Speedup |
|---|---:|---:|---:|---:|---:|---:|
| General / 2026-11-17 | 4,096 | 0.0471 | 0.2465 | 0.191x | 0.5343 | 0.0881x |
| General / 2026-11-17 | 32,768 | 2.594 | 1.310 | **1.98x** | 2.835 | 0.915x |
| General / 2026-11-17 | 131,072 | 40.455 | 5.914 | **6.84x** | 11.560 | **3.50x** |
| Optimized / 2026-11-18 | 4,096 | 0.004442 | 0.1961 | 0.0227x | 0.4331 | 0.0103x |
| Optimized / 2026-11-18 | 32,768 | 0.3320 | 1.049 | 0.316x | 2.269 | 0.146x |
| Optimized / 2026-11-18 | 131,072 | 5.873 | 5.060 | **1.16x** | 10.199 | 0.576x |
| Optimized / 2026-11-18 | 1,000,000 | 855 | Out of memory | — | Not timed | — |

Speedup is direct Numba time divided by FMM time; values above one favor FMM.
Among completed cases, FMM wins against optimized Numba only at 131,072 points
and `tol=1e-6`. At that size and `tol=1e-8`, optimized Numba is **1.74x faster**.
The crossover depends on geometry, tolerance, hardware and threading.

At one million points, the OS killed the first FMM warmup for lack of memory.
Subsequent checks with a 12 GiB memory limit failed at both tolerances; no FMM
speedup was measured. The direct median is **14.2 minutes**, rounded to seconds.

FMM required substantially more memory for this case: about **0.6 GiB** observed
for the Numba benchmark process, versus **17.6 GiB** for FMM when killed.
Much of the additional FMM storage holds multipole, local and plane-wave
expansions for the tree boxes; its size depends on tolerance and tree structure.

Historical direct entries are single full solves; reported new timings and historical
FMM timings are medians of three full solves after warmup, excluding compilation.
Completed solves compute every target velocity. At 131,072 points, sampled FMM
relative errors against
an independent direct sum are **3.63e-6** at `1e-6` and **5.00e-9** at `1e-8`;
native FMM tolerance does not guarantee relative accuracy of the assembled wall
field. The smaller comparison cases, **22 Python acceptance tests** and the full
MATLAB regression suite passed; the million-point FMM case remains incomplete.
See [methods, accuracy and provenance](docs/benchmarks.md#performance-compared-with-direct-numba-evaluation),
[guard dependency and thread controls](docs/python.md#parallel-proximity-guard),
and [OpenMP build/verification instructions](docs/dependencies.md#python-bindings-with-openmp).

### FMM3D versus STKFMM with a wall

[STKFMM](https://github.com/wenyan4work/STKFMM/tree/master)'s `StkWallFMM` class (on its
`master` branch) evaluates the same wall Stokeslet velocity with PVFMM. It runs
a Stokeslet FMM over physical and reflected sources plus two Laplace FMMs;
this repository's default path makes two FMM3D calls. STKFMM requires the wall
to be the lower face of its cubic box, with every point in the box's lower
half. Its output excludes viscosity.
[How it is called](docs/benchmarks.md#how-stkfmms-wall-kernel-is-called).

The comparison used the fixture and CPU of the Numba study above, with
**eight verified OpenMP threads for both codes**, on 2026-11-19. Times are
medians in seconds of three complete evaluations, each rebuilding all trees.
STKFMM order is not a tolerance, so each FMM3D tolerance is paired with the
STKFMM order of similar measured accuracy:

| Ns = Nt | FMM3D, tol 1e-6 | STKFMM, order 10 | FMM3D / STKFMM | FMM3D, tol 1e-8 | STKFMM, order 12 | FMM3D / STKFMM |
|---:|---:|---:|---:|---:|---:|---:|
| 4,096 | 0.178 | 0.209 | 0.86x | 0.413 | 0.330 | 1.25x |
| 32,768 | 1.056 | 0.932 | **1.13x** | 2.164 | 1.365 | **1.59x** |
| 131,072 | 4.703 | 3.626 | **1.30x** | 8.812 | 5.179 | **1.70x** |

Ratios above one mean STKFMM is faster. STKFMM is faster in every case except
N = 4,096 at `1e-6`, and its advantage grows with problem size and accuracy.

**Where the difference comes from.** These times cover the complete wall
evaluation, including each code's wall correction. Both codes build the wall
the same way: a free-space Stokeslet FMM over the physical and reflected
sources, plus Laplace FMMs for the harmonic correction. STKFMM splits the
force components slightly differently and uses one more Laplace evaluation.

Phase timings show that the speed difference comes from the free-space
Stokeslet FMM, not the wall correction. At 131,072 points, FMM3D's
free-space Stokes call alone (3.88 s at `1e-6`, 7.28 s at `1e-8`) takes
longer than STKFMM's entire wall evaluation (3.63 s and 5.18 s). FMM3D's
Laplace wall correction adds only 0.78–1.50 s. The comparison is therefore
mainly between STKFMM's PVFMM-based free-space FMM and FMM3D, including their
leaf sizes and builds.

Errors are against the independent direct Blake sum at 256 targets. At
131,072 points, the relative error is **4.90e-6** for FMM3D at `1e-6` versus
**9.01e-7** for STKFMM at order 10, and **5.87e-9** for FMM3D at `1e-8` versus
**2.92e-9** for STKFMM at order 12. STKFMM order 8 is faster still (2.78 s)
but only reaches an error of about 5e-5.

Most of FMM3D's time is its Stokes call: 7.28 s of 8.81 s at 131,072 points
and `1e-8`. STKFMM's timings exclude its one-time operator loading (0.6–5 s
per order) and use a tuned leaf size of `max_pts=500`; FMM3D ran with its
defaults. Reusing STKFMM's trees for new forces saves only 7–11%. Only one
geometry was measured, and other geometries may scale differently.

See [all orders, validation, caveats and provenance](docs/benchmarks.md#performance-compared-with-stkfmms-wall-fmm-2026-11-19)
and the [driver](benchmarks/compare_stkfmm_wall.py).

## Detailed documentation

- [MATLAB options, prepared geometry, fields and tests](docs/matlab.md)
- [Python interface and validation driver](docs/python.md)
- [Formulas, normalization and derivative conventions](docs/formulas.md)
- [Dependencies and tested system](docs/dependencies.md)
