function [u, info] = apply(geom, forces, opts)
%APPLY Apply forces to prepared geometry; FMM work is rebuilt on each call.
if nargin == 0, self_test(); u = []; info = []; return; end
total_timer=tic;
if nargin < 3, opts = struct(); end
if isfield(opts,'wall_height') && opts.wall_height ~= geom.wall_height
    error('blake:WallMismatch','Prepared wall height cannot be changed by apply.');
end
opts = blake.internal.options(opts);
[fields,info] = blake.internal.evaluate(geom,forces,opts,1);
u = fields.velocity;
info=blake.internal.finalize_timing(info,toc(total_timer),opts.method);
end

function self_test()
% Check the prepared interface against one-shot evaluation.
s = [0;0;1]; t = [0.2;0.3;2]; f = [1;2;3]; o.method = 'ggv-direct';
a = blake.apply(blake.prepare(s,t),f,o);
b = blake.velocity(s,f,t,o);
assert(norm(a-b) < 1e-14);
end
