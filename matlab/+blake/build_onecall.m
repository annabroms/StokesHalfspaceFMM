function build = build_onecall()
%BUILD_ONECALL Compile the optional GNU Fortran/MEX backend against selected FMM3D.
% Run blake.setup first. The external installation is read, never modified.
manifest=blake.internal.fmm.manifest();
root=fileparts(fileparts(fileparts(mfilename('fullpath'))));
native=fullfile(root,'native');
build_dir=fullfile(native,'build');
if ~isfolder(build_dir), mkdir(build_dir); end
archive=fullfile(fileparts(fileparts(manifest.stokes_path)),'lib-static','libfmm3d.a');
assert(isfile(archive),'blake:MissingFMM','Build the external FMM3D static library first.');
object=fullfile(build_dir,'blake_onecall.o');
command=['gfortran -O3 -fPIC -fopenmp -c ' quote(fullfile(native,'blake_onecall.f90')) ...
    ' -o ' quote(object)];
[status,message]=system(command);
assert(status==0,'blake:BuildFailed','Fortran compilation failed: %s',message);
output=fullfile(root,'matlab','+blake','+internal','+fmm','blake_onecall_mex');
clear blake.internal.fmm.blake_onecall_mex
mex('-compatibleArrayDims','CFLAGS=$CFLAGS -std=c99 -O3', ...
    'LDFLAGS=$LDFLAGS -Wl,--wrap=lfmm3d_', ...
    fullfile(native,'blake_onecall_mex.c'),object,archive, ...
    '-lgfortran','-lgomp','-lm','-lstdc++','-ldl','-output',output);
rehash
build=struct('archive',archive,'archive_sha256',checksum(archive), ...
    'fortran_sha256',checksum(fullfile(native,'blake_onecall.f90')), ...
    'gateway_sha256',checksum(fullfile(native,'blake_onecall_mex.c')), ...
    'mex_path',[output '.' mexext],'mex_sha256',checksum([output '.' mexext]), ...
    'fmm_source_revision',manifest.source_revision,'matlab',version, ...
    'compiler_flags','gfortran -O3 -fPIC -fopenmp');
[~,build.compiler]=system('gfortran --version');
fid=fopen(fullfile(build_dir,'build.json'),'w');
assert(fid>=0,'blake:BuildFailed','Cannot write build provenance.');
cleanup=onCleanup(@() fclose(fid));
fprintf(fid,'%s\n',jsonencode(build,PrettyPrint=true));
fprintf('Built %s\n',build.mex_path);
end

function text = quote(text)
% Quote a path for a read-only/build shell command.
text=['''' strrep(text,'''','''"''"''') ''''];
end

function value = checksum(path)
% Record the exact dependency or output binary, not just its source revision.
[status,value]=system(['sha256sum ' quote(path)]);
assert(status==0,'blake:BuildFailed','Cannot checksum %s.',path);
value=extractBefore(strtrim(value),' ');
end
