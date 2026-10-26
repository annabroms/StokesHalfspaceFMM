function [geom, prep_info] = prepare(sources, targets, wall_height, precheck_fmm)
%PREPARE Validate and reflect geometry; no FMM trees are cached.
if nargin == 0, self_test(); geom = []; prep_info=[]; return; end
timer=tic;
if nargin < 3, wall_height = 0; end
if nargin < 4, precheck_fmm=true; end
validateattributes(precheck_fmm,{'logical'},{'scalar'});
validateattributes(sources,{'double'},{'real','finite','nonsparse','2d','nrows',3});
validateattributes(targets,{'double'},{'real','finite','nonsparse','2d','nrows',3});
validateattributes(wall_height,{'double'},{'real','finite','scalar'});
if any(sources(3,:) <= wall_height) || any(targets(3,:) < wall_height)
    error('blake:InvalidGeometry','Sources must be above and targets on or above the wall.');
end
% Exact equality uses original coordinates, before any translation roundoff.
if ~isempty(sources) && ~isempty(targets) && ...
        ~isempty(intersect(sources.',targets.','rows'))
    error('blake:SingularInteraction','A physical source coincides with a target.');
end
sources(3,:) = sources(3,:) - wall_height;
targets(3,:) = targets(3,:) - wall_height;
if any(~isfinite(sources(:))) || any(~isfinite(targets(:)))
    error('blake:InvalidGeometry','Wall translation overflowed.');
end
% Recenter horizontal coordinates to reduce origin-dependent FMM cancellation.
points = [sources targets];
origin = zeros(2,1);
if ~isempty(points)
    origin = min(points(1:2,:),[],2)/2 + max(points(1:2,:),[],2)/2;
    sources(1:2,:) = sources(1:2,:) - origin;
    targets(1:2,:) = targets(1:2,:) - origin;
end
images = sources;
images(3,:) = -images(3,:);
augmented = [sources images targets];
span = 0;
if ~isempty(augmented), span = max(max(augmented,[],2)-min(augmented,[],2)); end
geom = struct('kind','blake.geometry.v1','sources',sources,'targets',targets, ...
    'images',images,'heights',sources(3,:),'wall_height',wall_height, ...
    'horizontal_origin',origin,'span',span,'ns',size(sources,2), ...
    'nt',size(targets,2),'physical_indices',1:size(sources,2), ...
    'image_indices',size(sources,2)+(1:size(sources,2)),'fmm_guard',[]);
prep_info=struct('geometry_seconds',toc(timer),'guard_seconds',0);
if precheck_fmm
    guard_timer=tic;
    geom.fmm_guard=blake.internal.fmm.guard(geom);
    prep_info.guard_seconds=toc(guard_timer);
end
prep_info.total_seconds=toc(timer);
end

function self_test()
% Check reflection and independent target counts without external dependencies.
g = blake.prepare([0;0;2],[1 2;0 0;1 3],1);
assert(g.ns == 1 && g.nt == 2 && g.images(3) == -1);
end
