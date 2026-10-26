function require()
%REQUIRE Check one coherent external installation before invoking its MEX.
paths={which('stfmm3d'),which('lfmm3d'),which('fmm3d')};
if any(cellfun(@isempty,paths)) || exist('fmm3d','file')~=3
    error('blake:MissingFMM','Run blake.setup with a compiled FMM3D MATLAB directory.');
end
folders=cellfun(@fileparts,paths,'UniformOutput',false);
if ~strcmp(folders{1},folders{2}) || ~strcmp(folders{1},folders{3})
    error('blake:MixedFMM','Stokes, Laplace and MEX paths must belong to one installation.');
end
end
