subroutine blake_onecall(ns, nt, source, force, charge_pn, dipole_pn, target, eps, velocity, pressure, ier) &
    bind(C, name='blake_onecall')
  ! Gimbutas--Greengard--Veerapaneni half-space representation, folded into four Laplace potentials.
  use iso_c_binding, only: c_int64_t, c_double
  implicit none
  integer(c_int64_t), intent(in) :: ns, nt
  real(c_double), intent(in) :: source(3,2*ns), force(3,2*ns), charge_pn(ns), dipole_pn(3,ns)
  real(c_double), intent(in) :: target(3,nt), eps
  real(c_double), intent(out) :: velocity(3,nt), pressure(nt)
  integer(c_int64_t), intent(out) :: ier
  real(c_double), allocatable :: y(:,:), x(:,:), charge(:,:), dipole(:,:,:), pot(:,:), grad(:,:,:)
  real(c_double) :: center(3), lower(3), upper(3)
  ! LFMM3D requires these arguments even when their outputs are disabled.
  real(c_double) :: unused_pot(4,1), unused_grad(4,3,1), unused_hess(4,6,1), unused_hesstarg(4,6,1)
  integer(c_int64_t) :: nd, nsource, ifcharge, ifdipole, iper, ifpgh, ifpghtarg, i, j, k
  integer :: allocation_status
  external lfmm3d

  ier=0
  velocity=0
  pressure=0
  if (ns==0 .or. nt==0) return
  nsource=2*ns
  allocate(y(3,nsource), x(3,nt), charge(4,nsource), dipole(4,3,nsource), &
      pot(4,nt), grad(4,3,nt), stat=allocation_status)
  if (allocation_status/=0) then
    ier=4
    return
  end if

  ! All coordinates use one center; target heights relative to the wall are not redefined.
  lower=min(minval(source,dim=2),minval(target,dim=2))
  upper=max(maxval(source,dim=2),maxval(target,dim=2))
  center=lower/2+upper/2
  do j=1,nsource
    y(:,j)=source(:,j)-center
    charge(1:3,j)=force(:,j)/2
    charge(4,j)=sum(y(:,j)*force(:,j))/2
  end do
  do j=1,nt
    x(:,j)=target(:,j)-center
  end do
  dipole=0

  ! With X=x-c, the original velocity is psi-X.grad(psi)+grad(psi4)
  ! minus (X3+c3)*grad(phi) plus e3*phi. Fold phi into psi3 and -c3*phi
  ! into psi4. The same reconstruction now gives the complete wall velocity.
  ! PN charges and dipoles live only on image sources (the second half).
  do j=1,ns
    k=ns+j
    charge(3,k)=charge(3,k)+charge_pn(j)
    charge(4,k)=charge(4,k)-center(3)*charge_pn(j)
    dipole(3,:,k)=dipole_pn(:,j)
    dipole(4,:,k)=-center(3)*dipole_pn(:,j)
  end do

  nd=4
  ifcharge=1
  ifdipole=1
  iper=0
  ifpgh=0
  ifpghtarg=2
  ! Exactly one FMM invocation; no source fields and no Laplace Hessians.
  call lfmm3d(nd,eps,nsource,y,ifcharge,charge,ifdipole,dipole,iper,ifpgh, &
      unused_pot,unused_grad,unused_hess,nt,x,ifpghtarg,pot,grad,unused_hesstarg,ier)
  if (ier/=0) return

  do k=1,nt
    do i=1,3
      velocity(i,k)=pot(i,k)+grad(4,i,k)-sum(x(:,k)*grad(1:3,i,k))
    end do
    ! Folding phi into psi3 also incorporates the PN pressure correction.
    pressure(k)=-2*(grad(1,1,k)+grad(2,2,k)+grad(3,3,k))
  end do
end subroutine blake_onecall
