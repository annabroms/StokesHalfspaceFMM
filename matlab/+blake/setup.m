function manifest = setup(fmm_directory)
%SETUP Add exactly one external FMM3D MATLAB directory to the current path.
if nargin == 0
    fmm_directory = '/home/anna/Dokument/FMM3D/FMM3D/matlab';
end
required = {'stfmm3d.m', 'lfmm3d.m', ['fmm3d.' mexext]};
for k = 1:numel(required)
    if ~isfile(fullfile(fmm_directory, required{k}))
        error('blake:MissingFMM', 'Missing %s in %s.', required{k}, fmm_directory);
    end
end
addpath(fmm_directory, '-begin');
manifest = blake.internal.fmm.manifest();
end
