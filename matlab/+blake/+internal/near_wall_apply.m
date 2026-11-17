function [fields, info] = near_wall_apply(geom, forces, opts, order, selected)
%NEAR_WALL_APPLY Directly replace complete target fields, keeping FMM on the rest.
if nargin==0, self_test(); fields=[]; info=[]; return; end
far_options=opts;
far_options.correct_near_wall=false;
far_options.diagnostics=false;
far_geometry=target_subset(geom,~selected);
[far_fields,info]=blake.internal.evaluate(far_geometry,forces,far_options,order);
t=tic;
near_geometry=target_subset(geom,selected);
near_fields=blake.internal.direct.blake_apply(near_geometry,forces,opts,order);
info.near_wall_direct_seconds=toc(t);
info.direct_seconds=info.direct_seconds+info.near_wall_direct_seconds;
assembly_timer=tic;
names=fieldnames(far_fields);
fields=struct();
for k=1:numel(names)
    name=names{k};
    if strcmp(name,'grad_u')
        fields.(name)=zeros(3,3,geom.nt);
        fields.(name)(:,:,~selected)=far_fields.(name);
        fields.(name)(:,:,selected)=near_fields.(name);
    else
        fields.(name)=zeros(size(far_fields.(name),1),geom.nt);
        fields.(name)(:,~selected)=far_fields.(name);
        fields.(name)(:,selected)=near_fields.(name);
    end
end
info.assembly_seconds=info.assembly_seconds+toc(assembly_timer);
info.ns=geom.ns; info.nt=geom.nt;
end

function subset = target_subset(geom, selected)
% Preserve translated coordinates and invalidate the guard for a changed target set.
subset=geom;
subset.targets=geom.targets(:,selected);
subset.nt=size(subset.targets,2);
subset.fmm_guard=[];
end

function self_test()
% Exercise all-direct replacement independently of compiled FMM dependencies.
g=blake.prepare([0 0.5;0 0.5;1e-6 1],[0;0;0],0,false);
f=[1 2;3 -1;2 1];
o=blake.internal.options(struct('correct_near_wall',true));
[a,info]=blake.internal.near_wall_apply(g,f,o,1,true);
b=blake.internal.direct.blake_apply(g,f,o,1);
assert(isequal(a,b) && info.fmm_call_count==0);
end
