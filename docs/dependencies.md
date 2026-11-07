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
checkout, then run `clear fmm3d` before `setpath` here. The setup path selects
`../FMM3D/FMM3D/matlab`, never the sibling `FMM3D_basic` installation. No
standalone `stfmm3d_laplace` wrapper is required.

## Tested system

The regression suite, validation driver and saved quick-profile benchmarks
were run on:

| Component | Tested configuration |
|---|---|
| Operating system | Ubuntu 22.04.5 LTS, Linux x86-64 |
| MATLAB | R2025a, version 25.1.0.2943329 |
| Processor | AMD Ryzen 7 PRO 8840HS, 8 cores / 16 logical CPUs |
| FMM3D source checkout | Commit `70455e020be1ff64445757cf71ae361f741ea282` |
| FMM thread settings | Separate runs with `OMP_NUM_THREADS=1` and `8` |

Other MATLAB releases and operating systems have not been validated here.
The source revision identifies the tested checkout; it does not by itself
establish which source produced a particular MEX binary.

## Reproducibility and interface limits

Each saved benchmark report records the resolved dependency paths, source
revision, MEX checksum, MATLAB version, hardware and thread settings. Generated
reports under `results/` are ignored by Git. Machine-specific installation
notes are kept separately in the ignored `docs/dependencies.local.md` file.

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
resolved wrappers, binary checksum and default backend; evaluation records
name the backend actually used. The historical tested-system table above
records the earlier validation environment, not a fingerprint of a newly
rebuilt MEX binary.

Prepared geometry caches a version-1 proximity assessment, not FMM data. Missing,
unchecked or differently versioned assessments are recomputed before evaluation.
There is no tree or expansion reuse, and no persistent global geometry cache.
