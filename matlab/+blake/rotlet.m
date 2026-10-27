function [u, info] = rotlet(sources, torques, targets, opts)
%ROTLET Complete no-slip wall velocity; see docs/singularities.md.
if nargin == 0
    selftest();
    u=[]; info=[];
    return
end
if nargin < 4, opts=struct(); end
[u,info]=blake.internal.singularities('rotlet',sources,torques,targets,opts);
end

function selftest()
% Exercise this family's actual wall and off-wall calculations.
blake.internal.singularity_test('rotlet');
end
