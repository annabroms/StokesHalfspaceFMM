function results = run_tests(require_fmm)
%RUN_TESTS Run all regressions; full acceptance rejects skipped FMM tests.
if nargin==0, require_fmm=true; end
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab'));
if require_fmm, blake.internal.fmm.require(); end
results=runtests(fileparts(mfilename('fullpath')));
if require_fmm
    assertSuccess(results);
else
    assert(~any([results.Failed]),'blake:TestsFailed','A regression failed.');
    fprintf('%d incomplete tests (FMM-dependent tests may be skipped).\n',sum([results.Incomplete]));
end
end
