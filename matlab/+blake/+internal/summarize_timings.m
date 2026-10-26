function summary = summarize_timings(samples)
%SUMMARIZE_TIMINGS Validate same-invocation timings and median each raw column.
if nargin==0, self_test(); summary=[]; return; end
names=samples.Properties.VariableNames;
if ismember('backend_call_seconds',names)
    calls=samples.backend_call_seconds;
else
    calls=samples.stokes_seconds+samples.laplace_seconds;
    assert(all(abs(calls-samples.fmm_evaluation_seconds)<=64*eps.*max(1,calls)), ...
        'blake:InvalidTiming','FMM components do not match combined times.');
end
slack=64*eps.*max(1,samples.total_seconds);
assert(all(calls>=0 & samples.overhead_seconds>=-slack), ...
    'blake:InvalidTiming','Negative call time or overhead.');
assert(all(abs(samples.total_seconds-calls-samples.overhead_seconds)<=slack), ...
    'blake:InvalidTiming','Timing values must describe the same invocation.');
summary=struct();
for k=1:numel(names)
    if endsWith(names{k},'_seconds')
        summary.(names{k})=median(samples.(names{k}));
    end
end
end

function self_test()
% Detect the difference between medians of sums and sums of medians.
a=table([1;9;9],[9;1;9],[10;10;18],[11;30;20],[1;20;2], ...
    'VariableNames',{'stokes_seconds','laplace_seconds','fmm_evaluation_seconds', ...
    'total_seconds','overhead_seconds'});
s=blake.internal.summarize_timings(a);
assert(s.fmm_evaluation_seconds==10 && s.overhead_seconds==2);
end
