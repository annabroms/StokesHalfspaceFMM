# Mathematical conventions and independent implementations

Coordinates have components in rows, points in columns. A force acts on the
fluid. The governing equations are `-grad(p) + mu*laplacian(u) + f*delta = 0`
and `div(u)=0`. Subtract wall height from both vertical coordinates; the
translated source height h is positive and translated target height z is nonnegative.
Recenter horizontal coordinates jointly for numerical stability. This changes
neither the wall nor the displacement vectors. Viscosity is a positive dynamic viscosity.

## Original sources checked

1. J. R. Blake, *A note on the image system for a stokeslet in a no-slip boundary*,
   Proc. Cambridge Philos. Soc. **70** (1971), 303–310,
   <https://doi.org/10.1017/S0305004100049902>.
   Original publisher PDF:
   <https://www.cambridge.org/core/services/aop-cambridge-core/content/view/S0305004100049902>.
   Checked the original equation (16), p. 307, including its pressure formula,
   and free-space normalizations (3–4), p. 304.
2. Z. Gimbutas, L. Greengard and S. Veerapaneni, *Simple and efficient
   representations for the fundamental solutions of Stokes flow in a half-space*,
   J. Fluid Mech. **776** (2015), R1, <https://doi.org/10.1017/jfm.2015.302>.
   Formula numbering below is the checked **arXiv v2** numbering,
   <https://arxiv.org/abs/1504.00102v2>, not an assertion about journal numbering.

## Blake reference (independent of GGV)

Write r=x-y, y*=(y1,y2,-h), R=x-y*, and m_j=1-2 delta_j3.
The reflection factor in Blake's (16), delta_jalpha delta_al - delta_j3 delta_3l,
reduces to m_j times differentiation with respect to R_j. The index alpha runs
over 1,2. Target velocity index is i, force index is j.

The implemented tensor is

```
B_ij = [T_ij(r)-T_ij(R) + 2 h^2 m_j D_ij(R) - 2 h m_j E_ij(R)] / (8 pi mu)
T_ij(r) = delta_ij/r + r_i r_j/r^3
D_ij(R) = delta_ij/R^3 - 3 R_i R_j/R^5
E_ij(R) = -delta_i3 R_j/R^3 + (delta_ij R_3 + R_i delta_j3)/R^3
          - 3 R_i R_3 R_j/R^5
```

Here m_j is not summed in the definitions; j is contracted with f_j when
applying the tensor. `blake_pair` uses explicit tensor entries. `blake_apply`
contracts each term directly with f or diag(1,1,-1)*f using bounded blocks.
The two routines do not call GGV helpers or each other for velocity.

Blake's pressure coefficient, independent of mu, is

```
P_j = [r_j/r^3 - R_j/R^3
       -2 h m_j (delta_3j/R^3 - 3 R_3 R_j/R^5)] / (4 pi).
```

Optional target derivatives in `blake_pair` are explicit analytic derivatives
of T, D and E. No symbolic derivatives, finite differences, or automatic
differentiation are used in production. `G(i,j,k)=dB(i,j)/dx_k`.
Direct pressure/gradient evaluation intentionally uses this readable reference.

## Half-space image representation

Equations (4–5) give S=T/(8 pi) and the free-space pressure kernel with 1/(4 pi).
Equation (13) gives f*=(f1,f2,-f3). Equation (14) uses 1/(4 pi R), and (15)
defines a dipole using a **source** derivative:

```
d . grad_y [1/(4 pi |x-y|)] = -d . grad_x [1/(4 pi |x-y|)]
                          = d . R/(4 pi R^3).
```

Equations (21–24) require physical Stokes strength f at y and strength -f*
at y*, followed by subtraction of the Papkovich–Neuber velocity. Use charge
q=f*_3=-f3 and dipole vector d=h*f* at y*. Summing source contributions:

```
phi = sum [q/R + d.R/R^3]/(4 pi)
g = grad(phi) = sum [-q R/R^3 + d/R^3 - 3 R(d.R)/R^5]/(4 pi)
v = sum S(x,y)f - sum S(x,y*)f*
u = [v - z*g + e3*phi]/mu
```

GGV (3) and (17) pair the PN velocity with pressure 2*dphi/dz. Since the
PN field is subtracted in (21), the complete fields are

```
p = p_Stokes - 2*g3
du_i/dx_j = [dv_i/dx_j - delta_j3*g_i - z*H_ij + delta_i3*g_j]/mu
traction = [-p I + mu (grad_u + grad_u^T)] normal
```

The harmonic Hessian is

```
H_ij = sum [q*(3 R_i R_j/R^5 - delta_ij/R^3)
            -3*(d_i R_j + d_j R_i + delta_ij*d.R)/R^5
            +15*(d.R)*R_i*R_j/R^7] / (4 pi).
```

Pressure is not divided by viscosity. Target gradients use velocity component
first, derivative direction second. FMM3D returns these indices in reverse
order, so its adapter explicitly transposes. Laplace packed Hessians are
`xx yy zz xy xz yz`; dipole Hessians are supported by the installed backend.
Pressure uses Stokes flag 2 and Laplace flag 2. Gradients and traction use flag
3 for both. The upstream velocity option uses Stokes flag 1 and Laplace flag 2,
with no source outputs requested through the MATLAB API.

## Target-only four-potential Stokes velocity

The default velocity adapter evaluates the same free-space Stokeslet using
four scalar charge potentials through the existing compiled Laplace FMM. It
also accepts arbitrary free-space sources, independent targets and signed forces
without introducing a wall or reflected sources itself.

For locations Y and strengths F, let G=1/(4*pi*|x-Y|). Define

```
psi_j = sum_n (F_jn/2) G(x,Y_n), j=1,2,3
psi_4 = sum_n (Y_n . F_n)/2 G(x,Y_n)
v_i = psi_i - sum_j x_j * d_i psi_j + d_i psi_4
```

Since d_i G=-(x_i-Y_i)/(4*pi*|x-Y|^3), the latter two terms combine
to r_i*(r.F)/(8*pi*r^3). Together with psi_i they give exactly the ordinary
unit-viscosity Stokeslet, including 1/(8*pi). Coordinates are jointly recentered
before constructing moments, reducing origin-dependent floating-point cancellation.

Use `nd=4`, four charge rows, source flag 0 and target flag 2. There are no
dipoles, source evaluations or target Hessians in this call. The adapter
reshapes gradients to density-by-derivative-by-target (4-by-3-by-Nt), reconstructs
velocity, and checks the exposed Laplace status. The harmonic correction remains
a separate charge/dipole call.

`stokes_backend='auto'` selects this adapter for velocity only. Pressure and
gradient requests retain upstream `stfmm3d`; explicit `laplace-velocity` with
such requests is rejected. No kernel normalization or physical conventions change.

## Errors, singularities and dimensional scaling

An exact physical source/target pair is singular and always rejected. Image
sources stay below the wall and are separate identities. No singular term is
silently omitted, no regularization is introduced, and no wall velocity is
forced to zero. A separate domain-relative FMM guard protects the backend's
near-coincidence exclusion; it is not a self-interaction prescription.

The diagnostic scale at each target is the sum over sources of the norms of
the physical Stokes term, reflected Stokes term, z*g_n and e3*phi_n, all divided
by mu. This provides a nonnegative constituent field scale for cancellation.
It is computed by independent direct kernels and costs O(Ns*Nt).

Maximum absolute error uses a pointwise Euclidean norm; global relative error
uses Frobenius norms. If the reference norm is at most 100*eps times the scale
norm, the global relative error is NaN. No pointwise velocity-relative error is
reported. Wall checks use absolute velocity divided by constituent scale.

When coordinates and wall height scale by s at fixed force and mu, every
velocity term scales as 1/s. Pressure and velocity gradients scale as 1/s^2.
Source height factors in d and the PN factor z must scale with the geometry.
