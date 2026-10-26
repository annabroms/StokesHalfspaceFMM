function [u, info] = velocity(sources, forces, targets, opts)
%VELOCITY Physical Stokes velocity above a horizontal no-slip wall.
% Arrays are 3-by-N, viscosity is opts.mu. See README for method options.
if nargin == 0, self_test(); u = []; info = []; return; end
total_timer=tic;
if nargin < 4, opts = struct(); end
opts = blake.internal.options(opts);
[geom,prep]=blake.prepare(sources,targets,opts.wall_height,strcmp(opts.method,'fmm'));
[fields,info] = blake.internal.evaluate(geom,forces,opts,1);
u = fields.velocity;
info.geometry_seconds=prep.geometry_seconds;
info.guard_seconds=info.guard_seconds+prep.guard_seconds;
info.guard_reused=false;
info=blake.internal.finalize_timing(info,toc(total_timer),opts.method);
end

function self_test()
% Verify the wall velocity without requiring FMM3D.
u = blake.velocity([0;0;1],[1;2;3],[0.2;0.3;0],struct('method','blake'));
assert(norm(u) < 1e-14);
end
