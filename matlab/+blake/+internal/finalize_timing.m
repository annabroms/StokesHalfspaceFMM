function info = finalize_timing(info, total_seconds, method)
%FINALIZE_TIMING Derive call and overhead totals from one public invocation.
if nargin==0, self_test(); info=[]; return; end
info.total_seconds=total_seconds;
info.timing_schema_version=2;
info.fmm_evaluation_seconds=0;
info.overhead_seconds=NaN;
if strcmp(method,'fmm')
    info.fmm_evaluation_seconds=info.stokes_seconds+info.laplace_seconds;
    info.overhead_seconds=total_seconds-info.fmm_evaluation_seconds;
end
end

function self_test()
% Ensure overhead uses a single total rather than a difference of medians.
info=struct('stokes_seconds',2,'laplace_seconds',1);
a=blake.internal.finalize_timing(info,4,'fmm');
assert(a.fmm_evaluation_seconds==3 && a.overhead_seconds==1);
end
