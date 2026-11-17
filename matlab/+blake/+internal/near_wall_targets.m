function [selected, relative_radius] = near_wall_targets(geom, tolerance, distance)
%NEAR_WALL_TARGETS Select ill-conditioned close targets using bounded cell searches.
if nargin==0, self_test(); selected=[]; relative_radius=[]; return; end
if nargin<3, distance=[]; end
validateattributes(tolerance,{'double'},{'real','finite','scalar','>',0,'<',1});
if ~isempty(distance)
    validateattributes(distance,{'double'},{'real','finite','scalar','positive'});
end
% eps*span/d estimates amplification in global-coordinate Stokes reconstruction.
relative_radius=min(2,max(64*eps,8*eps/tolerance));
selected=false(1,geom.nt);
if geom.ns==0 || geom.nt==0, return; end
if ~isfinite(geom.span) || geom.span<=0
    error('blake:FMMResolution','Geometry span must be finite and positive.');
end
if ~isempty(distance)
    relative_radius=max(64*eps,min(2,distance/geom.span));
end
sources=geom.sources/geom.span; targets=geom.targets/geom.span;
source_cells=floor(sources.'/relative_radius);
target_cells=floor(targets.'/relative_radius);
[cells,~,group]=unique(source_cells,'rows');
[~,permutation]=sort(group);
counts=accumarray(group,1);
ends=cumsum(counts); starts=[1;ends(1:end-1)+1];
% Above the wall, image distances cannot be less than physical distances.
for a=-1:1
    for b=-1:1
        for c=-1:1
            [found,index]=ismember(target_cells+[a b c],cells,'rows');
            hits=find(found.' & ~selected);
            for k=1:numel(hits)
                m=hits(k); cell_index=index(m);
                candidates=permutation(starts(cell_index):ends(cell_index));
                for n=1:512:numel(candidates)
                    subset=candidates(n:min(n+511,numel(candidates)));
                    separation=(sources(:,subset)-targets(:,m))/relative_radius;
                    if any(sum(separation.^2,1)<=1)
                        selected(m)=true;
                        break;
                    end
                end
            end
        end
    end
end
end

function self_test()
% Check selection and an explicit physical-distance override without FMM3D.
g=blake.prepare([0;0;1e-6],[0 0.4;0 0;0 1],0,false);
[selected,radius]=blake.internal.near_wall_targets(g,1e-9);
assert(isequal(selected,[true false]) && radius>0);
selected=blake.internal.near_wall_targets(g,1e-9,1e-7);
assert(~any(selected));
end
