# Wall rotlets, potential dipoles and oriented stresslets

All three routines compute physical velocity, with components in rows and points
in columns. Geometry, viscosity, coincidence errors and horizontal recentering
reuse the Stokeslet interface. There were **no existing free-space rotlet,
potential-dipole or stresslet routines** in this repository. The definitions below
therefore explicitly extend the existing Stokeslet normalization; no previous
stresslet parametrization is replaced.

```matlab
s = [0 0.4; 0 0.2; 1 0.8];
x = [0.1 0.8; 0.3 0.2; 0 0.7];
f = [1 -0.2; 0.4 0.7; -0.3 1];
o = struct('mu',1.7,'wall_height',0,'block_size',[256 512], ...
           'diagnostics',true);
[uR,infoR] = blake.rotlet(s,f,x,o);
[uD,infoD] = blake.potential_dipole(s,f,x,o);
o.normals = [0 1; 0 0; 1 0]; % unit SOURCE normals, 3-by-Ns
[uT,infoT] = blake.stresslet(s,f,x,o);
```

Sources lie strictly above the horizontal wall; targets lie on or above it.
Normals are source orientations here, unlike the **target** traction normals in
`blake.fields`. Arbitrary unit orientations are supported and are never silently
normalized. Their magnitude belongs in the strength. No quadrature weight is
inserted. Each source includes its complete wall correction individually.

## Definitions and coefficients

Let `r=x-y`, `h=y3-wall_height`, `z=x3-wall_height`, `M=diag(1,1,-1)`,
`R=x-y*`, and `y*=(y1,y2,wall_height-h)`. The free-space velocities are:

| Family | Free-space velocity | Independent coefficients per fixed source |
| --- | --- | --- |
| Rotlet | `cross(tau,r)/(8*pi*mu*norm(r)^3)` | 3 torque components |
| Potential dipole | `(d/r^3-3*r*(r.d)/r^5)/(8*pi*mu)` | 3 dipole moments |
| Oriented stresslet | `6*r*(r.g)*(r.n)/(8*pi*mu*r^5)` | 3 double-force components `g`, with prescribed unit `n` |

These are linear maps in the three strength components. The stresslet is also
bilinear in `g,n`; it is **not** an arbitrary symmetric tensor or a general
Stokeslet dipole. Its positive sign follows the cited paper. In particular, the
trace contribution when `g.n` is nonzero is retained. The torque is physical
force times length; `g` has force times length units and `d` has force times
length squared units. All velocities divide by dynamic viscosity. Thus, for fixed
strengths, uniform coordinate scaling about the wall gives powers -2, -3 and -2,
respectively. Other literature may absorb viscosity, reverse the double-layer
normal, or define a potential dipole with the opposite sign or a factor of two.

## Checked references

Z. Gimbutas, L. Greengard and S. Veerapaneni, *Simple and efficient representations
for the fundamental solutions of Stokes flow in a half-space*, JFM 776 (2015), R1,
[DOI](https://doi.org/10.1017/jfm.2015.302). Equation numbers here refer to the
[arXiv v2 PDF](https://arxiv.org/pdf/1504.00102v2), checked directly during
implementation: stresslet (7)-(9), harmonic dipole/quadrupole (15)-(16),
Papkovich–Neuber (17), stresslet wall correction (26)-(27), symmetric doublet
relation (28)-(32), and rotlet (33)-(35).

The potential dipole is derived below from the **entire** independent Blake tensor,
J. R. Blake (1971), equation (16), whose normalization is documented in
[formulas](formulas.md). It is not presented as a standalone potential-dipole wall
formula in the Gimbutas–Greengard–Veerapaneni paper.

## Computational sections and derivation

Every block computes physical and reflected separations once, then reuses inverse
radial powers. No full source-target matrix or tensor is assembled. Temporary
storage is proportional to the target/source block product. Work is O(Ns*Nt).
The final accumulation actually evaluates all wall components; none is zeroed.

For the following potentials, use `G=1/(4*pi*R)`,
`GD[a]=(a.R)/(4*pi*R^3)` and
`GQ[a,b]=(3*(a.R)*(b.R)/R^5-a.b/R^3)/(4*pi)`.
All image-plus-PN velocities have the form

```
u = (u_free - u_image - z*grad(phi) + e3*phi)/mu.
```

### Rotlet

The paper's oriented rotlet (33) contracts to
`[g*(n.r)-n*(g.r)]/(4*pi*r^3)`. Matching this with physical torque gives
`tau=2*cross(n,g)`. The image torque is the axial vector `tauI=-M*tau`.
Equation (35) then reduces to `phi=GD[cross(e3,tauI)]`.
The code evaluates the two cross products and the explicit dipole gradient
`[a/R^3-3*R*(a.R)/R^5]/(4*pi)`.
This includes the PN correction, not just a signed mirror rotlet.

### Oriented stresslet

Use `gI=M*g`, `nI=M*n`. Equations (26)-(27) give

```
u_image = 6*R*(R.gI)*(R.nI)/(8*pi*R^5)
phi = 2*(n.g)*GD[e3] + 2*h*GQ[nI,gI].
```

The explicit gradient of the quadrupole numerator is

```
grad(3*(R.a)*(R.b)/R^5 - a.b/R^3)
 = 3*(a*(R.b)+b*(R.a)+(a.b)*R)/R^5
   -15*R*(R.a)*(R.b)/R^7.
```

The dipole and quadrupole PN terms preserve the trace as well as the oriented
part. This distinction is needed for the independent reference below.

### Potential dipole

Write Blake's tensor, including its `1/(8*pi*mu)` factor, as

```
B_ij = [T_ij(r)-T_ij(R)+2*h^2*m_j*D_ij(R)-2*h*m_j*E_ij(R)]/(8*pi*mu)
D_ij = delta_ij/R^3-3*R_i*R_j/R^5
E_ij = partial_Rj T_i3
m_j = M_jj.
```

Define this potential-dipole tensor as `K_ij = (1/2)*Delta_y B_ij`.
In free space, `Delta T=2D`, so this fixes the stated sign and normalization.
For the image coordinates, `partial_yk R_l=-M_kk*delta_kl`, in particular
`partial_y3 R_3=+1`; `partial_y3 h=1`. Retaining every product-rule term gives

```
K_ij = [D_ij(r)-D_ij(R)
        +m_j*(2*D_ij(R)+2*h*partial_R3 D_ij(R)-2*partial_R3 E_ij(R))]
       /(8*pi*mu).
```

Here `Delta D=0`, `Delta E_ij=2*partial_R3 D_ij`. For example,
`Delta_y(2*h^2*D)=4D+8h*partial_R3 D` and
`Delta_y(-2h*E)=-4*partial_R3 E-4h*partial_R3 D`.
The production code expands both remaining first derivatives analytically.
There are no finite differences, symbolic evaluations or automatic derivatives
in this computation. Since `B(y,x_wall)=0` for every admissible `y`, its full
source Laplacian also vanishes there.

## Interfaces, self-tests and limits

All functions accept independent `3-by-Ns` sources and strengths and `3-by-Nt`
targets. Options reuse validation from the existing interface. The default and
only supported method is `blake` (blocked direct); explicitly requesting FMM or
GGV-direct raises `blake:UnsupportedBackend`. Only velocity is supported.
`tol` is accepted for option compatibility but does not control direct arithmetic.
`diagnostics=true` returns a conservative sum of componentwise absolute physical,
image and correction constituents in `info.constituent_scale`; otherwise that
field is zero. `info.total_seconds` measures the one-shot application.
Pressure, traction, target gradients and an FMM adapter for these new source
families are not implemented. External FMM3D is unchanged.

Every new callable MATLAB file has its own no-argument self-test and prints a
concise assertion-based pass summary:

```matlab
addpath('matlab');
blake.rotlet();
blake.potential_dipole();
blake.stresslet();
blake.internal.singularities();
blake.internal.singularity_test();
% Per-family metrics:
r = blake.internal.singularity_test('potential-dipole');
```

Tests restore the caller's RNG state. Wall tests use one and four sources,
three independent strength components, all nine stresslet strength/normal basis
products, random vectors, heights 0.08, 0.3, 1.2 and 3 above a wall at 0.35,
and targets directly beneath and laterally displaced from sources. The Euclidean
wall residual divides by a nonzero constituent/source-geometry scale, with a 300-eps budget for
cancellation and summation, rather than dividing by the zero exact velocity.

Off-wall checks use source finite differences of the existing independent Blake
pair evaluator. Rotlets use antisymmetric first derivatives, potential dipoles
use half the source Laplacian, and stresslets use symmetric first derivatives
**plus** the independently evaluated flux field required by (31)-(32).
Seven halved steps from 0.02 to 0.0003125 must converge quadratically.
Finite differences occur only in this validation file. Additional checks cover
near-source normalization, increasing-height free-space limits, target-difference
incompressibility, coefficient linearity, source additivity/blocking, coordinate
scaling, wall translation, viscosity, empty arrays and explicit coincidence errors.

Run the full existing acceptance suite from the repository root:

```bash
OMP_NUM_THREADS=1 matlab -batch "addpath('matlab','tests'); blake.setup; run_tests;"
```

## Measured validation (2026-10-27)

The full suite passed **24/24 tests**, including FMM-dependent tests; there were
no incomplete tests. Code Analyzer reported zero issues in all six changed/new
MATLAB files. The new kernels themselves require no external dependency.

Per-family self-tests above measured the following maxima across their wall
fixtures; FD errors are off-wall global relative errors at the smallest step:

| Family | Absolute wall velocity | Normalized wall residual | Independent FD error |
| --- | ---: | ---: | ---: |
| Rotlet | 0 (computed) | 0 (computed) | 7.263e-8 |
| Potential dipole | 5.219e-14 | 4.657e-16 | 2.857e-7 |
| Oriented stresslet | 5.322e-15 | 1.149e-16 | 1.113e-7 |

All seven-step reference studies decrease approximately fourfold per halving.
Normalized divergence errors were 8.989e-11, 1.833e-9 and 8.492e-10, respectively.
At the largest common vertical shift (1000), free-space relative errors were
2.234e-10, 4.205e-10 and 5.924e-7. The wall normalization uses the larger of the
constituent scale and `sum_n |strength_n|/(8*pi*mu*|x-y_n|^p)`, so even a basis
field whose individual constituents vanish has a positive geometric scale.

A separate small runtime measurement used seed 7281, 64 sources, 96 targets,
`mu=1.7`, wall height 0.35, blocks `[16 32]`, diagnostics enabled, independent
standard-normal horizontal positions, source heights `0.08+3*rand` above the wall,
13 wall targets and 83 targets at heights `0.2+rand`. Following one warm call per
family, one-shot application times (including validation and geometry) were:

| Family | Seconds | Absolute wall velocity | Constituent-normalized wall residual |
| --- | ---: | ---: | ---: |
| Rotlet | 0.012810 | 0 (computed) | 0 (computed) |
| Potential dipole | 0.007301 | 1.890e-15 | 1.846e-16 |
| Oriented stresslet | 0.005074 | 1.745e-16 | 3.412e-17 |

These are individual small-problem timings, not benchmark-profile statistics.
Direct arithmetic has no requested approximation tolerance (the inherited
`tol=1e-9` option is unused). Runs used MATLAB R2025a
`25.1.0.2943329`, Linux x86-64, `OMP_NUM_THREADS=1`; MATLAB reported
`maxNumCompThreads=8`. FMM-dependent existing tests used the installation at
`/home/anna/Dokument/FMM3D/FMM3D/matlab`, source revision
`c89238cfaf32e4c3eff4192065462b8ec968ed51`, MEX SHA-256
`c6fd530d68482b2eff6ecdbb509affd7cb5689cffd2782b0dd1d2518053eaba9`.
This provenance was queried from the active installation and supersedes older
installation snapshots for this validation run. No benchmark profile was run.
