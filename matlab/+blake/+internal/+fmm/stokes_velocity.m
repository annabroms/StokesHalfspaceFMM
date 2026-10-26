function [velocity, timing] = stokes_velocity(sources, forces, targets, tolerance)
%STOKES_VELOCITY Free-space, mu=1 Stokeslets via four Laplace charge densities.
% Internal inputs are validated by the caller. No wall or image construction.
if nargin==0, self_test(); velocity=[]; timing=[]; return; end
total_timer=tic;
ns=size(sources,2); nt=size(targets,2);
timing=struct('backend_call_seconds',0,'packing_seconds',0,'assembly_seconds',0);
velocity=zeros(3,nt);
if ns==0 || nt==0
    timing.total_seconds=toc(total_timer); return;
end
blake.internal.fmm.require();
timer=tic;
% Recenter all coordinates to avoid origin-dependent moment cancellation.
lower=min(min(sources,[],2),min(targets,[],2));
upper=max(max(sources,[],2),max(targets,[],2));
center=lower/2+upper/2;
Y=sources-center; X=targets-center;
src=struct('sources',Y,'nd',4, ...
    'charges',[forces/2; sum(Y.*forces,1)/2]);
timing.packing_seconds=toc(timer);
timer=tic;
[L,ier]=lfmm3d(tolerance,src,0,X,2);
timing.backend_call_seconds=toc(timer);
if ier~=0, error('blake:FMMFailure','Stokes Laplace backend returned ier=%d.',ier); end
timer=tic;
if numel(L.pottarg)~=4*nt || numel(L.gradtarg)~=12*nt
    error('blake:FMMFailure','Unexpected four-density Laplace output dimensions.');
end
potential=reshape(L.pottarg,4,nt);
gradient=reshape(L.gradtarg,4,3,nt);
if any(~isfinite(potential(:))) || any(~isfinite(gradient(:)))
    error('blake:FMMFailure','Nonfinite four-density Laplace output.');
end
for i=1:3
    velocity(i,:)=potential(i,:)+reshape(gradient(4,i,:),1,nt);
    for j=1:3
        velocity(i,:)=velocity(i,:)-X(j,:).*reshape(gradient(j,i,:),1,nt);
    end
end
if any(~isfinite(velocity(:)))
    error('blake:FMMFailure','Nonfinite reconstructed Stokes velocity.');
end
timing.assembly_seconds=toc(timer);
timing.total_seconds=toc(total_timer);
end

function self_test()
% Compare the four-potential representation against an explicit axial Stokeslet.
[u,t]=blake.internal.fmm.stokes_velocity([0;0;0],[0;0;1],[0;0;2],1e-12);
assert(norm(u-[0;0;1/(8*pi)])<1e-13);
assert(t.total_seconds>=t.backend_call_seconds);
end
