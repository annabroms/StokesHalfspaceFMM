function [u, info] = potential_dipole(sources, moments, targets, opts)
%POTENTIAL_DIPOLE Complete no-slip wall velocity; see docs/singularities.md.
if nargin == 0
    selftest();
    u=[]; info=[];
    return
end
if nargin < 4, opts=struct(); end
[u,info]=blake.internal.singularities('potential-dipole',sources,moments,targets,opts);
end

function selftest()
% Exercise this family's actual wall and off-wall calculations.
blake.internal.singularity_test('potential-dipole');
end
