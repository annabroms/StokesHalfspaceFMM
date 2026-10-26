# Wrapper optimization verification

Implemented timing schema 2, cached proximity assessments, and the four-potential
target-only Stokes velocity adapter on 2026-10-26.

Correctness validation with MATLAB R2025a and the unchanged selected FMM3D binary:

- Full suite: **22 passed, 0 failed, 0 incomplete**.
- MATLAB Code Analyzer: **zero findings in 31 MATLAB files**.
- New tests cover free-space Cartesian/random forces, arbitrary coordinates,
  singleton/empty arrays, unequal counts, and a 1200-source multipole fixture.
- Both fast velocity backends are checked against direct Stokeslets and the
  independent direct Blake evaluator. Existing source-height, wall-residual,
  dimensional-scaling, pressure, gradient and complete traction checks pass.
- Cache tests cover safe/unsafe assessments, changed forces, missing/unchecked/
  differently versioned caches, and direct evaluation of FMM-unsafe geometry.
- Synthetic sample tests distinguish the median combined call time from a sum
  of medians, and median overhead from a difference of medians. Per-call timing
  identities and nonnegative overhead are checked as correctness properties.

`compare_stokes_backends` is a standalone **free-space** performance and accuracy
test, excluded from `run_tests`. It compares ordinary Stokeslet sources and
independent targets, without a wall, image sources or harmonic correction.
Its code was analyzed but the comparison itself was **not executed**.

No benchmark profiles, tolerance sweeps, block studies or figure-generation
drivers were executed during this change. No new performance measurements or
speedup claims are made. Existing saved result artifacts were preserved.

The new adapter uses the existing compiled `lfmm3d` interface and reports its
status explicitly. The external FMM3D installation and MFS repositories were
not edited. Pressure and derivative requests continue to use upstream Stokes.
