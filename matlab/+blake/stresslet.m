function [u, info] = stresslet(sources, strengths, targets, opts)
%STRESSLET Complete no-slip wall velocity; see docs/singularities.md.
if nargin == 0
    selftest();
    u=[]; info=[];
    return
end
if nargin < 4, opts=struct(); end
[u,info]=blake.internal.singularities('stresslet',sources,strengths,targets,opts);
end

function selftest()
% Exercise this family's actual wall and off-wall calculations.
blake.internal.singularity_test('stresslet');
end
