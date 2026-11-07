function manifest = setup(fmm_directory)
%SETUP Add exactly one external FMM3D MATLAB directory to the current path.
% An explicit argument takes priority over the FMM3D_MATLAB environment variable.
% Without either, look beside this repository in FMM3D/matlab or FMM3D/FMM3D/matlab.
if nargin == 0
    fmm_directory = getenv('FMM3D_MATLAB');
    if isempty(fmm_directory)
        repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
        parent = fileparts(repo_root);
        fmm_directory = fullfile(parent,'FMM3D','matlab');
        nested_directory = fullfile(parent,'FMM3D','FMM3D','matlab');
        if isfile(fullfile(nested_directory,'stfmm3d.m'))
            fmm_directory = nested_directory;
        end
    end
end
required = {'stfmm3d.m', 'lfmm3d.m', ['fmm3d.' mexext]};
for k = 1:numel(required)
    if ~isfile(fullfile(fmm_directory, required{k}))
        error('blake:MissingFMM', ...
            'Missing %s in %s. Pass blake.setup a directory or set FMM3D_MATLAB.', ...
            required{k}, fmm_directory);
    end
end
addpath(fmm_directory, '-begin');
manifest = blake.internal.fmm.manifest();
end
