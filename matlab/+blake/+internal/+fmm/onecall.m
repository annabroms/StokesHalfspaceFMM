function [fields,timing] = onecall(geom,data,tolerance,mu,order)
%ONECALL Fold the PN correction into four potentials in the optional Fortran MEX.
if nargin==0, self_test(); fields=[]; timing=[]; return; end
if order>2
    error('blake:UnsupportedBackend','one-call supports velocity and pressure, not gradients or traction.');
end
if exist(which('blake.internal.fmm.blake_onecall_mex'),'file')~=3
    error('blake:MissingOneCall','Run blake.build_onecall after selecting FMM3D with blake.setup.');
end
timing=struct('guard_seconds',0,'guard_reused',false,'strength_seconds',0, ...
    'assembly_seconds',0,'stokes_seconds',0,'laplace_seconds',0, ...
    'stokes_backend','folded-laplace','stokes_error_flag_exposed',true, ...
    'fmm_call_count',0,'laplace_source_flag',0,'laplace_target_flag',2);
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
timing.strength_seconds=toc(timer);
% Time the complete native call from MATLAB, including packing and reconstruction.
timer=tic;
[u,p,stats]=blake.internal.fmm.blake_onecall_mex( ...
    sources,data.stokes,data.charges,data.dipoles,geom.targets,tolerance);
timing.laplace_seconds=toc(timer);
if stats(1)~=0
    error('blake:FMMFailure','One-call Laplace FMM returned ier=%d.',stats(1));
end
timing.fmm_call_count=double(stats(2));
timing.laplace_source_flag=double(stats(3));
timing.laplace_target_flag=double(stats(4));
timer=tic;
fields=struct('velocity',u/mu);
if order>=2, fields.pressure=p; end
% Native reconstruction is already the complete half-space field.
timing.assembly_seconds=toc(timer);
end

function self_test()
% Check bulk and raw wall velocities against the independent direct Blake tensor.
s=[0;0;1]; f=[1;2;3]; x=[0.2 0.3;0.4 0.1;0 2];
g=blake.prepare(s,x); d=blake.internal.halfspace.strengths(g,f);
[a,t]=blake.internal.fmm.onecall(g,d,1e-12,1,2);
b=blake.fields(s,f,x,struct('method','blake','requested',{{'pressure'}}));
assert(norm(a.velocity-b.velocity,'fro')<1e-12 && norm(a.pressure-b.pressure)<1e-12);
assert(t.fmm_call_count==1 && t.laplace_source_flag==0 && t.laplace_target_flag==2);
end
