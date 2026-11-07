% Validate the comparison refactor and portable dependency selection.
root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
cd(root);
addpath('matlab','tests','benchmarks');
initial = blake.setup;
dependencyDirectory = fileparts(initial.stokes_path);
originalEnvironment = getenv('FMM3D_MATLAB');
environmentCleanup = onCleanup(@() setenv('FMM3D_MATLAB',originalEnvironment));

% A configured directory must override the default sibling installation.
configuredDirectory = fullfile(root,'native','build','selected_fmm_for_path_test');
setenv('FMM3D_MATLAB',configuredDirectory);
configured = blake.setup;
% MATLAB canonicalizes the configured symlink to its physical directory.
assert(strcmp(fileparts(configured.stokes_path),dependencyDirectory));

% An explicit argument must override even an invalid environment setting.
setenv('FMM3D_MATLAB',fullfile(root,'native','build','missing_fmm_for_path_test'));
explicit = blake.setup(dependencyDirectory);
assert(strcmp(fileparts(explicit.stokes_path),dependencyDirectory));
missingWasRejected = false;
try
    blake.setup;
catch exception
    assert(strcmp(exception.identifier,'blake:MissingFMM'));
    missingWasRejected = true;
end
assert(missingWasRejected);

% setpath must resolve its own location when called from another directory.
setenv('FMM3D_MATLAB',configuredDirectory);
cd(tempdir);
run(fullfile(root,'setpath.m'));
assert(strcmp(fileparts(which('stfmm3d')),dependencyDirectory));
cd(root);
fprintf('Portable setup checks passed.\n');

files = {'benchmarks/compare_fmm_calls.m','matlab/+blake/setup.m', ...
    'setpath.m','examples/minimal_example.m'};
for fileIndex = 1:numel(files)
    issues = checkcode(files{fileIndex},'-id');
    if ~isempty(issues)
        disp(issues);
    end
    assert(isempty(issues));
end
fprintf('Code Analyzer: zero findings in the four changed MATLAB files.\n');

results = run_tests;
assert(all([results.Passed]) && ~any([results.Incomplete]));

% The saved full sweep supplies the exact pre-refactor accuracy/reciprocity tables.
previousRecord = jsondecode(fileread('native/build/matlab_timing_validation.json'));
previous = load([previousRecord.scaling_stem '.mat'],'report');
report = compare_fmm_calls(struct('sizes',[1024 4096], ...
    'ratios',[0.25 1 4],'repetitions',2,'save',true));
assert(isequaln(report.accuracy,previous.report.accuracy));
assert(isequaln(report.reciprocity,previous.report.reciprocity));
assert(height(report.timings)==12 && height(report.samples)==48);
assert(isequal(report.timings.Properties.VariableNames, ...
    previous.report.timings.Properties.VariableNames));
assert(isequal(report.samples.Properties.VariableNames, ...
    previous.report.samples.Properties.VariableNames));
assert(all(isfinite(report.samples.total_seconds)));
assert(all(report.samples.total_seconds>=report.samples.fmm_seconds));
assert(all(report.samples.fmm_calls(strcmp(report.samples.backend,'one-call'))==1));
assert(all(report.samples.fmm_calls(strcmp(report.samples.backend,'two-call'))==2));
saved = load([report.output_stem '.mat'],'report');
assert(isequaln(saved.report,report));
for suffix = {'accuracy','reciprocity','timings','samples'}
    assert(isfile([report.output_stem '_' suffix{1} '.csv']));
end
fprintf('Readability regression passed: exact field/reciprocity tables, unchanged schema, 48 samples.\n');
fprintf('FINAL_SUITE %d passed, %d failed, %d incomplete.\n', ...
    sum([results.Passed]),sum([results.Failed]),sum([results.Incomplete]));
