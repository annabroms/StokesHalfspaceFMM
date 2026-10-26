% A horizontal no-slip wall; separate source and target arrays, in physical units.
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'matlab'));
blake.setup('/home/anna/Dokument/FMM3D/FMM3D/matlab');
sources=[0 0.5;0 0.2;1 0.7];
forces=[1 -0.3;0 0.4;0.2 1];
targets=[0.1 0.4 0.7;0.2 0.3 0.1;0 0.5 1.2];
opts=struct('method','fmm','mu',1.5,'tol',1e-9);
[velocity,info]=blake.velocity(sources,forces,targets,opts);
disp(velocity); disp(info);
% Repeated force applications reuse geometry preparation, but rebuild FMM trees.
geom=blake.prepare(sources,targets,0);
velocity2=blake.apply(geom,2*forces,opts);
assert(norm(velocity2-2*velocity,'fro')<1e-10);
