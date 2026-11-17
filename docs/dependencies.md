# Dependencies and tested system

## Requirements

- **MATLAB**. The implementation and tests have been validated with R2025a.
- **[FMM3D](https://github.com/flatironinstitute/FMM3D)** with its MATLAB
  interfaces compiled for your MATLAB version and platform. The FMM methods
  require `stfmm3d.m`, `lfmm3d.m`, and the platform-specific `fmm3d` MEX binary.
  Follow the [FMM3D installation instructions](https://fmm3d.readthedocs.io/en/latest/install.html).

The direct Blake and direct half-space image methods require only MATLAB. No additional
MATLAB toolbox is required. FMM3D is an external dependency and is not bundled
with this repository.

## Setup

From the repository root, run `setpath`. The supplied script adds this
repository's public MATLAB interfaces, examples, benchmarks and tests, and
looks for a sibling checkout at `../FMM3D/matlab` or `../FMM3D/FMM3D/matlab`,
relative to this repository's location. No username or home-directory path is
required. Set `FMM3D_MATLAB` to select another installation for all no-argument
setup calls:

```matlab
setenv('FMM3D_MATLAB', '/path/to/FMM3D/matlab');
setpath;
```

To select an installation explicitly for a setup call, use:

```matlab
addpath('matlab');
manifest = blake.setup('/path/to/FMM3D/matlab');
```

Select one compiled FMM3D installation explicitly. The Stokes wrapper, Laplace
wrapper and MEX binary must resolve to the same directory. Avoid recursively
adding directories containing several FMM3D installations.

Default velocity evaluation calls the updated `stfmm3d` with source flag 0
and target flag 1. Its internal four-potential Laplace call uses flags `(0,2)`:
no source fields or Hessians. The separate harmonic correction uses Laplace
flags `(0,2)` as well. Pressure requests also avoid Hessians; velocity-gradient
and traction requests require them. The optional `laplace-velocity` backend is
retained only as an explicitly selected MATLAB reference for comparisons.

Rebuild the updated FMM3D MATLAB interface with `make matlab` from its own
checkout, then run `clear fmm3d` before `setpath` here. Use `FMM3D_MATLAB` or
an explicit setup argument to select the rebuilt installation. No standalone
`stfmm3d_laplace` wrapper is required.

## Tested system

The regression suite, validation driver and saved quick-profile benchmarks
were run on:

| Component | Tested configuration |
|---|---|
| Operating system | Ubuntu 22.04.5 LTS, Linux x86-64 |
| MATLAB | R2025a, version 25.1.0.2943329 |
| Processor | AMD Ryzen 7 PRO 8840HS, 8 cores / 16 logical CPUs |
| FMM3D | Compiled MATLAB Stokes and Laplace interfaces |
| FMM thread settings | Separate runs with `OMP_NUM_THREADS=1` and `8` |

Other MATLAB releases and operating systems have not been validated here.

## Reproducibility and interface limits

Saved benchmark reports record dependency metadata, MATLAB version, hardware
and thread settings. Generate reports for your installation using the
[benchmark drivers](benchmarks.md). The measurements documented here include
their geometry, tolerance and execution settings.

The current adapter rebuilds FMM trees on each evaluation. The inspected
Stokes MATLAB wrapper does not expose its internal error code; the Laplace
wrapper does. The adapter checks returned shapes and finite values and includes
a proximity check for the backend's small-distance exclusion. See the README
and [formula conventions](formulas.md) for the numerical details.


## Velocity backend optimization

The updated external Stokes driver selects internal Laplace derivative flags
from the requested Stokes outputs. The default `auto` backend now resolves to
`upstream`, using that driver for velocities as well as derivative requests.
The older MATLAB four-potential adapter remains available explicitly as
`laplace-velocity` for the standalone comparison benchmark. It is no longer
needed to avoid unnecessary Hessians in normal velocity evaluation.

The `stokes_error_flag_exposed` field is false for the default Stokes wrapper
and true for the legacy `laplace-velocity` adapter. Manifests identify the
selected dependency and default backend; evaluation records name the backend
actually used. The tested-system table above describes the historical
validation environment.

Prepared geometry caches a version-1 proximity assessment, not FMM data. Missing,
unchecked or differently versioned assessments are recomputed before evaluation.
There is no tree or expansion reuse, and no persistent global geometry cache.

## Python guard dependency

The default Python FMM proximity guard requires Numba in addition to NumPy and
`fmm3dpy`; the `fmm` packaging extra declares both accelerated dependencies.
Install `.[fmm]` in the intended environment, preserving the selected OpenMP
FMM3D build. Direct-only evaluation still requires just NumPy. The measured
configuration uses Numba 0.65.1. Set `NUMBA_NUM_THREADS=8` alongside
`OMP_NUM_THREADS=8` for matched guard/FMM limits. See the
[guard behavior and first-call compilation](python.md#parallel-proximity-guard).

## Python bindings with OpenMP

OpenMP support in the Fortran source does not by itself enable threading in
Python wheels. The Meson Python build must compile the Fortran sources with
`-fopenmp` and link the extensions against the OpenMP runtime. On the tested
Linux/GCC system, the following builds a wheel from a disposable copy of the
selected FMM3D checkout, then installs it into the user Python environment:

```bash
python3 -m venv --system-site-packages .venv-fmm-openmp
.venv-fmm-openmp/bin/python -m pip install --upgrade pip meson-python
.venv-fmm-openmp/bin/python -m pip wheel --no-deps --no-build-isolation \
    -Csetup-args=-Dfortran_args=-fopenmp \
    -Csetup-args=-Dfortran_link_args=-fopenmp \
    -Csetup-args=-Dc_link_args=-fopenmp \
    -Ccompile-args=-j4 --wheel-dir wheels /path/to/disposable/FMM3D
python3 -m pip install --user --no-deps --force-reinstall wheels/fmm3dpy-*.whl
```

The build environment must contain NumPy and have `gfortran`, a C compiler and
Ninja available. Use a wheel directory containing only the intended build.
For an isolated runtime, install the wheel into that environment instead of
using `--user`. The measured rebuild used the existing FMM3D source revision
`0e13b3deb5e2cb35f9fe691464c00e2700208ad5`; no source changes were required.
The source checkout and MATLAB installation were left unchanged.

Start a fresh Python process after installation. Set `OMP_NUM_THREADS` before
importing the bindings, and verify both the imported package and its runtime:

```bash
OMP_NUM_THREADS=8 python3 - <<'PYOPENMP'
import ctypes
from pathlib import Path
import fmm3dpy

print(fmm3dpy.__file__)
for name in ("stfmm3d_fortran", "lfmm3d_fortran"):
    extension = next(Path(fmm3dpy.__file__).parent.glob(name + "*.so"))
    query = ctypes.CDLL(str(extension)).omp_get_max_threads
    query.argtypes = []
    query.restype = ctypes.c_int
    print(extension.name, "OpenMP maximum threads:", query())
PYOPENMP
```

Both extensions should report eight for this command. `ldd` should also list
`libgomp`, and `nm -D` should expose OpenMP runtime references. The query verifies
the configured thread limit rather than the number of threads active in every
FMM phase; use separate-process timing runs to measure the achieved speedup.
Package version `2.1.0` is unchanged by rebuilding, so retain the wheel or binary
hashes with performance results. See [the Numba comparison](benchmarks.md#performance-compared-with-direct-numba-evaluation).
