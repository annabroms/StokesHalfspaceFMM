function [result, info] = fields(sources, forces, targets, opts)
%FIELDS Velocity plus optional pressure, gradient and complete traction.
% opts.requested={'velocity','pressure','gradient','traction'};
% opts.normals is 3-by-Nt, unit length; grad_u(i,j,m)=du_i/dx_j.
if nargin==0, self_test(); result=[]; info=[]; return; end
total_timer=tic;
if nargin<4, opts=struct(); end
opts=blake.internal.options(opts);
order=1;
if ismember('pressure',opts.requested), order=2; end
if any(ismember(opts.requested,{'gradient','traction'})), order=3; end
traction=ismember('traction',opts.requested);
if traction
    validateattributes(opts.normals,{'double'}, ...
        {'real','finite','2d','nrows',3,'ncols',size(targets,2)});
    if any(abs(vecnorm(opts.normals)-1)>1e-12)
        error('blake:InvalidNormals','Traction normals must have unit length.');
    end
end
[geom,prep]=blake.prepare(sources,targets,opts.wall_height,strcmp(opts.method,'fmm'));
[result,info]=blake.internal.evaluate(geom,forces,opts,order);
info.geometry_seconds=prep.geometry_seconds;
info.guard_seconds=info.guard_seconds+prep.guard_seconds;
info.guard_reused=false;
if traction
    t=tic;
    result.traction=zeros(3,geom.nt);
    for m=1:geom.nt
        G=result.grad_u(:,:,m);
        stress=-result.pressure(m)*eye(3)+opts.mu*(G+G.');
        result.traction(:,m)=stress*opts.normals(:,m);
    end
    elapsed=toc(t); info.assembly_seconds=info.assembly_seconds+elapsed;
end
info=blake.internal.finalize_timing(info,toc(total_timer),opts.method);
end

function self_test()
% Check complete direct traction against the independent Blake derivative.
s=[0;0;1]; f=[1;2;3]; x=[0.2;0.4;0];
o=struct('method','blake','requested',{{'traction'}},'normals',[0;0;1]);
a=blake.fields(s,f,x,o); o.method='ggv-direct'; b=blake.fields(s,f,x,o);
assert(norm(a.traction-b.traction)<1e-13);
end
