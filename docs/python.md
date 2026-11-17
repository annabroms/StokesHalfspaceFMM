# Python Blake velocity

The Python implementation evaluates velocity above a horizontal no-slip wall.
The accelerated representation is due to **Z. Gimbutas, L. Greengard and
S. Veerapaneni**, *Simple and efficient representations for the fundamental
solutions of Stokes flow in a half-space*, JFM 776 (2015), R1,
[doi:10.1017/jfm.2015.302](https://doi.org/10.1017/jfm.2015.302).
The independent reference implements J. R. Blake's original tensor, equation
(16), [doi:10.1017/S0305004100049902](https://doi.org/10.1017/S0305004100049902).
See the shared [formula conventions](formulas.md).

## Environment and usage

Use Python 3.10 or newer with NumPy. Accelerated evaluation additionally needs
Numba and the existing compiled `fmm3dpy` installation. No MFS package or MATLAB bridge is
used. Nothing in this package builds or modifies the external FMM3D repository.
Select the interpreter that imports the intended installation:

```bash
python3 -c 'import sys, numpy, fmm3dpy; print(sys.executable); print(numpy.__version__); print(fmm3dpy.__file__)'
```

From the repository root, set `PYTHONPATH=python`, or install this repository's
package into an existing environment. A root-level project definition supplies
the package metadata; the distribution is `stokes-halfspace-fmm`, imported as
`blake`. FMM3D and Numba are optional for direct-only use; installing `.[fmm]` declares
both accelerated dependencies.

```python
import numpy as np
import blake

sources = np.array([[0, .5], [0, .2], [1, .7]])
forces = np.array([[1, -.3], [0, .4], [.2, 1]])
targets = np.array([[.1, .4, .7], [.2, .3, .1], [0, .5, 1.2]])

u, info = blake.velocity(sources, forces, targets,
                         mu=1.5, tol=1e-9, return_info=True)
reference = blake.velocity(sources, forces, targets, mu=1.5, method="blake")

geom = blake.prepare(sources, targets, wall_height=0)
same = blake.apply(geom, forces, mu=1.5, tol=1e-9)
tensor = blake.blake_pair(sources[:, :1], targets[:, 1:2], mu=1.5)
```

Inputs have components first, points second, including singletons `(3, 1)` and
empties `(3, 0)`. Real numeric inputs are copied into float64 storage; complex,
nonfinite and incorrectly shaped arrays are rejected. Sources must be strictly
above `wall_height`; targets may lie on the wall. `mu` is positive. Forces act
on the fluid, with no implicit quadrature weights. Returned velocity is physical
velocity, including division by viscosity. SI inputs produce m/s.

Defaults are `method="fmm"`, `mu=1`, `wall_height=0`, `tol=1e-9`,
`block_size=(256, 512)`, `correct_near_wall=False`, `near_wall_distance=None`,
`diagnostics=False`, `return_info=False`.
Block sizes are target/source counts for direct work and diagnostics.
The Python interface currently supports velocity only.

## Parallel proximity guard

Nonempty FMM evaluations use Numba occupied bins by default. Construction is
compiled and serial; independent target checks run in parallel. The index uses
linear storage and is rebuilt on each application. Physical and reflected
sources retain the inclusive `64*eps*span` exclusion radius and explicit
`FMMResolutionError`; no dense source-by-target matrix is formed.

Set both thread limits before starting Python, for example:

```bash
PYTHONPATH=python OMP_NUM_THREADS=8 NUMBA_NUM_THREADS=8 python3
```

`OMP_NUM_THREADS` controls the threaded FMM3D bindings, and `NUMBA_NUM_THREADS`
controls the guard. Both need enabled native threading support. The first
nonempty guard call may compile Numba kernels; subsequent calls reuse compiled
code, including an on-disk cache when available. Warm benchmark results exclude
that first-call cost. `info["guard_backend"]` identifies `"numba-bins"`, and
`info["guard_seconds"]` includes normalization, bin construction and querying.
Numba is imported lazily, so direct evaluation and empty FMM evaluations do not
require it. Missing Numba raises an explicit dependency error.

## Two-call evaluation

After translating the wall to zero, source height is `h`, target height is `z`,
and reflected quantities are `y*=(y1,y2,-h)`, `f*=(f1,f2,-f3)`.

1. Call `fmm3dpy.stfmm3d` with combined sources `[y,y*]` and Stokeslet
   strengths `[f,-f*]`, using `ifppreg=0`, `ifppregtarg=1`, `nd=1`.
2. Call `fmm3dpy.lfmm3d` at image sources with `charges=f*[2,:]` and
   `dipvec=h*f*`, using `pg=0`, `pgt=2`, `nd=1`.
3. Assemble `u=(v-z*grad(phi)+e3*phi)/mu` from the returned target fields.

### Optional near-wall correction

```python
u, info = blake.velocity(sources, forces, targets, tol=1e-12,
                         correct_near_wall=True, return_info=True)
```

The flag selects close targets using geometry and replaces their **complete**
velocity by the independent, blocked direct Blake sum over **all** sources.
Only the remaining targets are sent to the two FMM calls. If every target is
selected, there are no FMM calls or imports. Exact coincidences still raise;
computed wall residuals are never overwritten. The direct method accepts the
flag as a no-op. The default FMM behavior is unchanged.

Let `L=geom.span`, `d` be distance to the nearest physical source, and `eps` be
float64 machine epsilon. Selection uses
`d/L <= max(64*eps, 8*eps/tol)`. A radius of `2*L` suffices to select everything
when `tol <= 4*eps`. Above the wall, images are never closer than physical
sources. An occupied-cell search uses linear storage and bounded candidate
blocks; it does not build a source-by-target distance matrix. Selection is
independent of forces, viscosity and coordinate scale. Despite the flag's
name, it also catches very close source/target pairs above the wall.

The criterion reflects possible `eps*L/d` amplification of roundoff when
Stokes velocity is reconstructed from global-coordinate potentials. The factor
8 is conservative guidance, **not a guaranteed error bound**. Smaller `tol`
selects more targets. It cannot detect every cancellation in the final field,
and direct evaluation has its own roundoff floor near a zero velocity.

Set `near_wall_distance` to a positive distance in the same units as the
coordinates to override the automatic radius; `None` keeps the automatic rule.
For example, `near_wall_distance=1e-3*geom.span` provides an explicit relative
cutoff with prepared geometry. The effective radius is at least `64*eps*L`;
it is capped at `2*L`, which already covers every possible interaction. The
distance option has no effect unless correction and the FMM method are enabled.

Correction costs O(Ns*Ncorrected) direct work in bounded blocks. It can be cheap
for a few close targets and costly for a substantial fraction of the target
set. `info` reports `corrected_target_count`, `near_wall_relative_radius`,
`near_wall_distance` (the effective physical cutoff), `near_wall_detection_seconds`
and `near_wall_direct_seconds`; `direct_seconds`
also includes that direct work. `total_seconds` includes detection and correction.
Diagnostics retain their separate, optional O(Ns*Nt) cost.
See the [measured speed and accuracy comparison](near-wall-correction.md).

No extra `4*pi` or `8*pi` multiplier is applied: these normalizations are
already in the wrappers. `dipvec` denotes the source derivative of the harmonic
Green function. Both `ier` status codes, field presence, shapes and finite
values are checked. See the [official Python wrapper documentation](https://fmm3d.readthedocs.io/en/latest/python.html).

Each nonempty evaluation makes exactly two public wrapper calls. The internal
Stokes driver implementation and derivative work depend on the selected FMM3D
build. No Hessian is requested through the Python interface. Prepared geometry
reuses translation and reflection, but the proximity guard runs on each FMM
application, and neither FMM trees nor expansions are reused.

## Direct reference and exceptional cases

The [pair tensor](../python/blake/direct.py) computes explicit Blake tensor
entries; its blocked velocity contraction implements the original tensor
formula separately. Neither calls the Papkovich–Neuber assembly. Workspaces
are bounded by the supplied block sizes, with no full interaction matrix.

Exact physical source/target coincidences raise `SingularInteractionError`,
even for zero forces. Coincidences are checked in original coordinates before
translation. Duplicated sources contribute additively. Empty inputs preserve
output shape and require no FMM import or calls.

An occupied-bin proximity guard raises `FMMResolutionError` for physical or
reflected sources within `64*eps*span` of a target. It prevents the backend's
small-distance exclusion from silently dropping an interaction. Direct
evaluation remains available for noncoincident near pairs. With the correction
enabled, selected targets bypass FMM and its exclusion guard; all other targets
retain the guard. Overflow and
nonfinite outputs raise explicit errors; no regularization is introduced.

## Diagnostics and validation

`diagnostics=True` adds `info["constituent_scale"]`: the sum of the norms of
the individual physical Stokes, reflected Stokes, `z*grad(phi)` and `e3*phi`
terms, divided by `mu`. It costs O(Ns*Nt) direct work in bounded blocks and is
disabled by default. Call `blake.metrics(u, reference, scale)` for maximum
pointwise absolute error, global Frobenius relative error, and maximum
constituent-normalized error. Relative error is NaN for a negligible reference,
including a zero wall reference. The validation JSON encodes it as null.

```bash
PYTHONPATH=python OMP_NUM_THREADS=1 python3 python/tests/run_tests.py
PYTHONPATH=python OMP_NUM_THREADS=1 python3 examples/validate_velocity.py --output results/python_validation.json
```

Full acceptance exits unsuccessfully if any FMM tests are unavailable. For
development without FMM3D, `python/tests/run_tests.py --direct-only` allows
those tests to remain explicitly incomplete. It still runs FMM tests if the
dependency is available. NumPy is required in both modes.

Computational entry points have deterministic no-argument self-tests, for
example `blake.blake_pair()`, `blake.direct_velocity()`, `blake.prepare()`,
`blake.velocity()`, `blake.constituent_scale()` and `blake.metrics()`.
`blake.fmm.fmm_velocity()` tests the compiled dependency and fails if absent.
The correctness suite also checks both status codes and two-call dispatch.

The validation driver tests 1,200-source wrapper normalization, four geometries
over three requested tolerances, Cartesian and random forces, exact wall and
near-wall targets, independent direct fields, pair reciprocity and distributed
reciprocity on unequal point clouds. It records actual wrapper/extension paths,
interpreter and package versions, thread setting, geometry, tolerance and single-evaluation
timings. These small correctness cases do not establish FMM performance on
large systems. It runs no benchmark profile.

The empirical constituent-normalized budget is `100*tol+1e-12`; bulk relative
error uses `100*tol+1e-11` on these well-conditioned bulk fixtures. Neither is
a mathematical guarantee. Cancellation can set a tolerance-independent floor,
particularly immediately below shallow sources. All three wall components are
computed and reported without being overwritten. See the
[measured verification record](python-validation.md).

## Reading the implementation

Start with the public calls in [evaluate.py](../python/blake/evaluate.py).
Geometry preparation validates and translates the coordinates; application
then selects the direct or FMM evaluator. The numerical modules use `r` for
physical source-to-target displacement and `r_image` for image displacement.
Blocked pair arrays have axes `(component, target, source)`.

| Module | Purpose |
|---|---|
| [geometry.py](../python/blake/geometry.py) | Coordinate checks, reflection and the FMM proximity guard |
| [fmm.py](../python/blake/fmm.py) | Strength packing, the two FMM calls and physical velocity assembly |
| [near_wall.py](../python/blake/near_wall.py) | Optional close-target selection and geometry subsets |
| [direct.py](../python/blake/direct.py) | Independent Blake pair tensor and blocked force contraction |
| [diagnostics.py](../python/blake/diagnostics.py) | Constituent magnitudes and error metrics |
| [validation.py](../python/blake/validation.py) | Separate wrapper, field and reciprocity checks |
