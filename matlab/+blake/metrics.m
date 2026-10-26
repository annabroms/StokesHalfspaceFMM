function out = metrics(value, reference, scale)
%METRICS Absolute, global relative and constituent-normalized velocity error.
if nargin==0, self_test(); out=[]; return; end
validateattributes(reference,{'double'},{'size',size(value)});
validateattributes(scale,{'double'},{'real','finite','nonnegative','numel',size(value,2)});
pointwise=vecnorm(value-reference);
out.max_absolute=max([0 pointwise]);
out.global_relative=NaN;
if norm(reference,'fro')>100*eps*norm(scale)
    out.global_relative=norm(value-reference,'fro')/norm(reference,'fro');
end
ratio=zeros(size(pointwise)); active=scale>0;
ratio(active)=pointwise(active)./reshape(scale(active),1,[]);
ratio(~active & pointwise>0)=Inf;
out.constituent_normalized=max([0 ratio]);
end

function self_test()
% Ensure exact-zero wall references never produce a false relative metric.
a=blake.metrics(zeros(3,1),zeros(3,1),1);
assert(isnan(a.global_relative) && a.max_absolute==0);
end
