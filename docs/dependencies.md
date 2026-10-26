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
expects the external FMM3D MATLAB directory at `../FMM3D/FMM3D/matlab`.

For a different installation layout, use:

```matlab
addpath('matlab');
manifest = blake.setup('/path/to/FMM3D/matlab');
```

Select one compiled FMM3D installation explicitly. The Stokes wrapper, Laplace
wrapper and MEX binary must resolve to the same directory. Avoid recursively
adding directories containing several FMM3D installations.

Default velocity evaluation uses four batched Laplace charge potentials and
target gradients for Stokeslets, plus separate Laplace charge/dipole potentials
and gradients for the half-space correction. The optional `upstream` backend
uses Stokes target velocities instead. The optional pressure, velocity-gradient and
traction interfaces additionally use Stokes pressure/gradients and Laplace
target Hessians. These capabilities were verified in the tested installation.

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

The installed Stokes source driver requests internal Laplace source and target
Hessians even for a public target-velocity-only request. The new MATLAB adapter
bypasses that path using the same external binary's four-density Laplace call
with source flag 0 and target flag 2. This removes unnecessary derivative and
source-field work without editing or rebuilding FMM3D. It is an algorithmic work
reduction; no measured speedup is claimed until a new timing comparison is run.

The adapter exposes and checks Laplace ier for the Stokes velocity calculation.
The `stokes_error_flag_exposed` field is true for `laplace-velocity` and false
for the upstream path. Pressure and target derivatives retain the upstream
limitation. Manifests list available backends; evaluation/sample records name
which backend actually ran. Dependency source and binary provenance are unchanged.

Prepared geometry caches a version-1 proximity assessment, not FMM data. Missing,
unchecked or differently versioned assessments are recomputed before evaluation.
There is no tree or expansion reuse, and no persistent global geometry cache.
