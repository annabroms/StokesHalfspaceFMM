function [stokes, harmonic, timing] = evaluate(geom, data, tolerance, order, backend)
%EVALUATE Target-only Stokes and harmonic fields, using cached geometry checks.
if nargin==0, self_test(); stokes=[]; harmonic=[]; timing=[]; return; end
if nargin<5, backend='auto'; end
if strcmp(backend,'auto')
    backend='upstream';
    if order==1, backend='laplace-velocity'; end
end
if order>1 && strcmp(backend,'laplace-velocity')
    error('blake:UnsupportedBackend','laplace-velocity supports velocity only; use auto or upstream.');
end
blake.internal.fmm.require();
timing=struct('guard_seconds',0,'guard_reused',false, ...
    'strength_seconds',0,'assembly_seconds',0,'stokes_backend',backend, ...
    'stokes_error_flag_exposed',strcmp(backend,'laplace-velocity'));
cached=isfield(geom,'fmm_guard') && isstruct(geom.fmm_guard) ...
    && isscalar(geom.fmm_guard) && isfield(geom.fmm_guard,'version') ...
    && isequal(geom.fmm_guard.version,1) && geom.fmm_guard.checked;
if cached
    assessment=geom.fmm_guard; timing.guard_reused=true;
else
    timer=tic; assessment=blake.internal.fmm.guard(geom);
    timing.guard_seconds=toc(timer);
end
if ~assessment.safe, error('blake:FMMResolution','%s',assessment.message); end
timer=tic;
sources=[geom.sources geom.images];
l=struct('sources',geom.images,'charges',data.charges,'dipoles',data.dipoles);
if strcmp(backend,'upstream'), s=struct('sources',sources,'stoklet',data.stokes); end
 timing.strength_seconds=toc(timer);
if strcmp(backend,'laplace-velocity')
    [stokes.velocity,t]=blake.internal.fmm.stokes_velocity(sources,data.stokes,geom.targets,tolerance);
    timing.stokes_seconds=t.backend_call_seconds;
    timing.strength_seconds=timing.strength_seconds+t.packing_seconds;
    timing.assembly_seconds=timing.assembly_seconds+t.assembly_seconds;
else
    timer=tic; A=stfmm3d(tolerance,s,0,geom.targets,order);
    timing.stokes_seconds=toc(timer);
end
timer=tic;
[L,ier]=lfmm3d(tolerance,l,0,geom.targets,max(2,order));
timing.laplace_seconds=toc(timer);
if ier~=0, error('blake:FMMFailure','Harmonic Laplace FMM returned ier=%d.',ier); end
timer=tic; nt=geom.nt;
if strcmp(backend,'upstream'), stokes.velocity=reshape(A.pottarg,3,nt); end
harmonic.potential=reshape(L.pottarg,1,nt);
harmonic.gradient=reshape(L.gradtarg,3,nt);
if order>=2, stokes.pressure=reshape(A.pretarg,1,nt); end
if order>=3
    % FMM3D grad(derivative,velocity,target) -> public grad(velocity,derivative,target).
    stokes.gradient=permute(reshape(A.gradtarg,3,3,nt),[2 1 3]);
    packed=reshape(L.hesstarg,6,nt);
    indices=[1 4 5;4 2 6;5 6 3];
    harmonic.hessian=reshape(packed(indices(:),:),3,3,nt);
end
names=fieldnames(stokes);
for k=1:numel(names)
    if any(~isfinite(stokes.(names{k})(:)))
        error('blake:FMMFailure','Nonfinite Stokes output.');
    end
end
names=fieldnames(harmonic);
for k=1:numel(names)
    if any(~isfinite(harmonic.(names{k})(:)))
        error('blake:FMMFailure','Nonfinite harmonic output.');
    end
end
timing.assembly_seconds=timing.assembly_seconds+toc(timer);
end

function self_test()
% Compare the adapter with explicit free-space kernels on a small geometry.
g=blake.prepare([0;0;1],[0.2 0.3;0.4 0.1;0 2]);
d=blake.internal.halfspace.strengths(g,[1;2;3]);
[a,h]=blake.internal.fmm.evaluate(g,d,1e-12,3);
b=blake.internal.direct.stokes([g.sources g.images],d.stokes,g.targets,[2 2],3,false);
l=blake.internal.direct.laplace(g.images,d.charges,d.dipoles,g.targets,[2 2],3,false);
assert(norm(a.velocity-b.velocity,'fro')<1e-12);
assert(norm(a.gradient(:)-b.gradient(:))<1e-12);
assert(norm(h.hessian(:)-l.hessian(:))<1e-12);
end
