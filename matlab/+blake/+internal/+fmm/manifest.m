function info = manifest()
%MANIFEST Record source and binary provenance separately; no source checkout edits.
blake.internal.fmm.require();
info=struct('matlab',version,'computer',computer, ...
    'stokes_path',which('stfmm3d'),'laplace_path',which('lfmm3d'), ...
    'mex_path',which('fmm3d'),'omp_num_threads',getenv('OMP_NUM_THREADS'), ...
    'matlab_max_threads',maxNumCompThreads, ...
    'stokes_error_flag_exposed',false,'tree_reuse',false, ...
    'timing_schema_version',2,'velocity_default_backend','laplace-velocity', ...
    'available_stokes_backends',{{'upstream','laplace-velocity'}});
root=fileparts(fileparts(info.stokes_path));
% Quote filesystem paths before passing them to read-only shell commands.
quote = @(s) ['''' strrep(s,'''','''"''"''') ''''];
[status,revision]=system(['git -C ' quote(root) ' rev-parse HEAD']);
info.source_revision='unknown';
if status==0, info.source_revision=strtrim(revision); end
[status,checksum]=system(['sha256sum ' quote(info.mex_path)]);
info.mex_sha256='unknown';
if status==0, info.mex_sha256=extractBefore(strtrim(checksum),' '); end
[~,info.cpu]=system('lscpu');
[~,info.memory]=system('free -b');
[~,info.os]=system('uname -a');
end
