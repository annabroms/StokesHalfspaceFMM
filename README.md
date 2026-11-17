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

Use Python 3.10 or newer with NumPy and the compiled `fmm3dpy` interface.
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

Both languages evaluate wall Stokeslet velocity. MATLAB additionally provides:

- Pressure, velocity gradients and complete traction through `blake.fields`.
- Wall **rotlets**, **potential dipoles** and **stresslets** through
  `blake.rotlet`, `blake.potential_dipole` and `blake.stresslet`, using blocked
  direct evaluation.

See [MATLAB fields](docs/matlab.md#pressure-gradients-and-traction) and
[other kernel definitions and tests](docs/singularities.md).

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

## Detailed documentation

- [MATLAB options, prepared geometry, fields and tests](docs/matlab.md)
- [Python interface and validation driver](docs/python.md)
- [Formulas, normalization and derivative conventions](docs/formulas.md)
- [Dependencies and tested system](docs/dependencies.md)
