function [fields, info] = evaluate(geom, forces, opts, order)
%EVALUATE Dispatch kernels while preserving physical units and timing boundaries.
if nargin==0, self_test(); fields=[]; info=[]; return; end
timer=tic;
if ~isstruct(geom) || ~isscalar(geom) || ~isfield(geom,'kind') || ...
        ~strcmp(geom.kind,'blake.geometry.v1')
    error('blake:InvalidGeometry','Use geometry returned by blake.prepare.');
end
validateattributes(forces,{'double'},{'real','finite','nonsparse','2d','nrows',3,'ncols',geom.ns});
if strcmp(opts.method,'fmm') && order>1 && strcmp(opts.stokes_backend,'laplace-velocity')
    error('blake:UnsupportedBackend','laplace-velocity supports velocity only; use auto or upstream.');
end
info=struct('method',opts.method,'requested_tolerance',opts.tol, ...
    'ns',geom.ns,'nt',geom.nt,'geometry_seconds',0,'strength_seconds',0, ...
    'guard_seconds',0,'stokes_seconds',0,'laplace_seconds',0, ...
    'direct_seconds',0,'assembly_seconds',0,'diagnostics_seconds',0, ...
    'stokes_error_flag_exposed',false,'tree_reused',false, ...
    'guard_reused',false,'stokes_backend','none');
if geom.ns==0 || geom.nt==0
    fields=struct('velocity',zeros(3,geom.nt));
    if order>=2, fields.pressure=zeros(1,geom.nt); end
    if order>=3, fields.grad_u=zeros(3,3,geom.nt); end
    if opts.diagnostics, info.constituent_scale=zeros(1,geom.nt); end
    info=blake.internal.finalize_timing(info,toc(timer),opts.method); return;
end
t=tic; data=blake.internal.halfspace.strengths(geom,forces);
info.strength_seconds=toc(t);
if strcmp(opts.method,'blake')
    t=tic; fields=blake.internal.direct.blake_apply(geom,forces,opts,order);
    info.direct_seconds=toc(t);
else
    if strcmp(opts.method,'fmm')
        [s,l,timing]=blake.internal.fmm.evaluate(geom,data,opts.tol,order,opts.stokes_backend);
        info.guard_seconds=timing.guard_seconds;
        info.guard_reused=timing.guard_reused;
        info.stokes_backend=timing.stokes_backend;
        info.stokes_error_flag_exposed=timing.stokes_error_flag_exposed;
        info.strength_seconds=info.strength_seconds+timing.strength_seconds;
        info.assembly_seconds=timing.assembly_seconds;
        info.stokes_seconds=timing.stokes_seconds;
        info.laplace_seconds=timing.laplace_seconds;
    else
        t=tic;
        s=blake.internal.direct.stokes([geom.sources geom.images],data.stokes, ...
            geom.targets,opts.block_size,order,false);
        info.stokes_seconds=toc(t); t=tic;
        l=blake.internal.direct.laplace(geom.images,data.charges,data.dipoles, ...
            geom.targets,opts.block_size,order,false);
        info.laplace_seconds=toc(t);
    end
    t=tic;
    fields=blake.internal.halfspace.assemble(s,l,geom.targets(3,:),opts.mu,order);
    info.assembly_seconds=info.assembly_seconds+toc(t);
end
if opts.diagnostics
    % Independent O(Ns*Nt) scale calculation is opt-in, never hidden in FMM timing.
    t=tic;
    a=blake.internal.direct.stokes(geom.sources,forces,geom.targets,opts.block_size,1,true);
    b=blake.internal.direct.stokes(geom.images,data.reflected,geom.targets,opts.block_size,1,true);
    h=blake.internal.direct.laplace(geom.images,data.charges,data.dipoles, ...
        geom.targets,opts.block_size,1,true);
    info.constituent_scale=(a.scale+b.scale+abs(geom.targets(3,:)).*h.gradient_scale ...
        +h.potential_scale)/opts.mu;
    info.diagnostics_seconds=toc(t);
end
names=fieldnames(fields);
for k=1:numel(names)
    if any(~isfinite(fields.(names{k})(:)))
        error('blake:NumericalRange','Field overflowed; rescale the geometry and forces.');
    end
end
info=blake.internal.finalize_timing(info,toc(timer),opts.method);
end

function self_test()
% Verify independent direct formulations on a nonzero bulk velocity.
g=blake.prepare([0;0;1],[0.2;0.4;2]); f=[1;2;3];
o=blake.internal.options(struct('method','blake'));
a=blake.internal.evaluate(g,f,o,1); o.method='ggv-direct';
b=blake.internal.evaluate(g,f,o,1);
assert(norm(a.velocity-b.velocity)<1e-14);
end
