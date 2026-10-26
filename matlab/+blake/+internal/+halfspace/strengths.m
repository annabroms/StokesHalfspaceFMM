function data = strengths(geom, forces)
%STRENGTHS GGV (13), (21)-(24): minus reflected Stokes, plus h*f* dipole.
if nargin == 0, self_test(); data = []; return; end
reflected = forces;
reflected(3,:) = -reflected(3,:);
data = struct('reflected',reflected,'stokes',[forces -reflected], ...
    'charges',reflected(3,:),'dipoles',reflected.*geom.heights);
end

function self_test()
% Lock the independent Stokes-image and harmonic-dipole signs.
d = blake.internal.halfspace.strengths(blake.prepare([0;0;2],[1;0;0]),[1;2;3]);
assert(isequal(d.stokes(:,2),[-1;-2;3]) && isequal(d.dipoles,[2;4;-6]));
end
