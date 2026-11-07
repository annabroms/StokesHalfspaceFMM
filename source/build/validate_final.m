cd('/home/anna/Dokument/StokesHalfspaceFMM');
addpath('matlab','tests','benchmarks'); blake.setup;
files={'matlab/+blake/build_onecall.m','matlab/+blake/+internal/+fmm/onecall.m', ...
 'matlab/+blake/+internal/options.m','matlab/+blake/+internal/evaluate.m', ...
 'matlab/+blake/+internal/+fmm/manifest.m','tests/test_onecall.m','benchmarks/compare_fmm_calls.m'};
for k=1:numel(files)
    issues=checkcode(files{k},'-id');
    for j=1:numel(issues), fprintf('%s line %d: %s\n',files{k},issues(j).line,issues(j).message); end
    assert(isempty(issues));
end
timer=tic; tests=run_tests; suite_seconds=toc(timer);
% Repeat correctness with the enlarged reciprocity clouds, not the completed timing sweep.
text=evalc('validation=compare_fmm_calls(struct(''sizes'',1024,''ratios'',1,''repetitions'',1,''save'',true));');
disp(validation.accuracy); disp(validation.reciprocity);
load('results/tp9e61bc22_0a97_4335_912d_fba894d4f0df_fmm_calls.mat','report');
record=struct('passed',sum([tests.Passed]),'failed',sum([tests.Failed]), ...
 'incomplete',sum([tests.Incomplete]),'suite_seconds',suite_seconds, ...
 'accuracy',table2struct(validation.accuracy),'reciprocity',table2struct(validation.reciprocity), ...
 'timings',table2struct(report.timings),'configuration',report.configuration, ...
 'metadata',report.metadata,'scaling_stem',report.output_stem,'validation_stem',validation.output_stem);
fid=fopen('native/build/final_validation.json','w');
fprintf(fid,'%s',jsonencode(record)); fclose(fid);
fprintf('FINAL_SUITE %d passed %d failed %d incomplete in %.4fs\n', ...
 record.passed,record.failed,record.incomplete,suite_seconds);
