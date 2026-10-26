# Numerical development

This repository is independent of the MFS repositories. Do not add an MFS
dependency or edit the external FMM3D installation.

Preserve component-first 3-by-N arrays, physical velocity units and explicit
coincidence errors. Keep direct Blake velocity and derivative references
independent of the half-space image representation with its Papkovich–Neuber
correction. Attribute the representation to all three authors. Never zero computed wall residuals.
Do not introduce dense Ns-by-Nt interaction storage or finite differences in
production kernels. Maintain independently runnable no-argument self-tests for
computational functions and one-line purpose comments on local helpers.

From the repository root, validate numerical edits with:

```bash
OMP_NUM_THREADS=1 matlab -batch "addpath('matlab','tests'); blake.setup; run_tests;"
```

FMM-unavailable tests are incomplete, not passing. `run_tests(false)` permits
direct-only development. Run benchmark profiles only when requested: even the
quick profile collects five repetitions over 45 cases and can take minutes.
Use Code Analyzer on changed MATLAB files. Record measured accuracy and runtime
with the geometry, tolerance, thread count and dependency provenance.
