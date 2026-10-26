function assessment = guard(geom)
%GUARD Assess geometry; with no output, raise an error on an unsafe pair.
if nargin==0, self_test(); assessment=[]; return; end
tau=64*eps*geom.span;
assessment=struct('version',1,'checked',true,'safe',true,'threshold',tau, ...
    'source_index',0,'target_index',0,'source_kind','','message','');
if geom.ns==0 || geom.nt==0, return; end
if ~isfinite(tau) || tau<=0
    assessment.safe=false;
    assessment.message='Geometry span cannot be resolved by this FMM backend.';
    if nargout==0, error('blake:FMMResolution','%s',assessment.message); end
    return;
end
sources=[geom.sources geom.images]; targets=geom.targets;
origin=min([sources targets],[],2);
% Wider cells leave room for quantization roundoff at domain-sized coordinates.
source_bins=floor((sources-origin).'/(2*tau));
target_bins=floor((targets-origin).'/(2*tau));
[bins,~,group]=unique(source_bins,'rows');
[~,permutation]=sort(group);
counts=accumarray(group,1);
ends=cumsum(counts); starts=[1;ends(1:end-1)+1];
for a=-1:1
    for b=-1:1
        for c=-1:1
            [found,index]=ismember(target_bins+[a b c],bins,'rows');
            hits=find(found);
            for k=1:numel(hits)
                m=hits(k); bin=index(m);
                candidates=permutation(starts(bin):ends(bin));
                % Check candidates in bounded chunks even for duplicate sources.
                for n=1:512:numel(candidates)
                    subset=candidates(n:min(n+511,numel(candidates)));
                    distances=(sources(:,subset)-targets(:,m))/tau;
                    close=find(sum(distances.^2,1)<=1,1);
                    if ~isempty(close)
                        source_index=subset(close); category='physical';
                        if source_index>geom.ns
                            category='reflected'; source_index=source_index-geom.ns;
                        end
                        assessment.safe=false;
                        assessment.source_index=source_index; assessment.target_index=m;
                        assessment.source_kind=category;
                        assessment.message=sprintf( ...
                            'Target %d is within %.3g of %s source %d. Use a direct method.', ...
                            m,tau,category,source_index);
                        if nargout==0, error('blake:FMMResolution','%s',assessment.message); end
                        return;
                    end
                end
            end
        end
    end
end
end

function self_test()
% Ensure image proximity is guarded even though the physical source is distinct.
g=blake.prepare([0 1;0 1;1e-16 1],[0;0;0],0,false);
try
    blake.internal.fmm.guard(g);
catch exception
    assert(strcmp(exception.identifier,'blake:FMMResolution')); return;
end
error('blake:SelfTest','The proximity guard failed.');
end
