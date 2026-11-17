function report = compare_near_wall_rule(repetitions)
%COMPARE_NEAR_WALL_RULE Time full MATLAB velocity calls at cutoff=1e-3*span.
if nargin<1, repetitions=3; end
validateattributes(repetitions,{'double'},{'scalar','integer','>=',3});
root=fileparts(fileparts(mfilename('fullpath')));
data=jsondecode(fileread(fullfile(root,'results','near_wall_rule_fixtures.json')));
report=struct('matlab',version,'omp_threads',getenv('OMP_NUM_THREADS'), ...
    'matlab_max_threads',maxNumCompThreads,'stokes_wrapper',which('stfmm3d'), ...
    'laplace_wrapper',which('lfmm3d'),'fmm_extension',which('fmm3d'), ...
    'backend','default two-call','repetitions',repetitions, ...
    'relative_cutoff',data.relative_cutoff,'tol',data.tol,'mu',data.mu, ...
    'wall_height',data.wall_height, ...
    'timing_scope',['complete public velocity call including geometry, exclusion guard, ' ...
        'selection, packing, two FMM calls, assembly and direct correction; ' ...
        'one warm-up per path; alternating order; reference checks excluded'], ...
    'cases',{{}});
for k=1:numel(data.cases)
    report.cases{k}=compare_case(data.cases(k),data,repetitions);
    destination=fullfile(root,'results','near_wall_rule_matlab.json');
    fid=fopen(destination,'w');
    assert(fid>=0,'Cannot open comparison output.');
    cleanup=onCleanup(@() fclose(fid));
    fprintf(fid,'%s\n',jsonencode(report));
    clear cleanup;
end
end

function result = compare_case(input, settings, repetitions)
% Warm each path and retain repeated full-call timings on identical exported inputs.
s=input.sources; f=input.forces; x=input.targets;
o=struct('tol',settings.tol,'mu',settings.mu,'near_wall_distance',input.near_wall_distance);
fprintf('MATLAB: starting %s (%d/%d)\n',input.case,input.ns,input.nt);
for flag=[false true]
    o.correct_near_wall=flag;
    blake.velocity(s,f,x,o);
end
samples=zeros(2,repetitions); phases=cell(2,repetitions); values=cell(1,2);
for repetition=1:repetitions
    order=[1 2];
    if mod(repetition,2)==0, order=[2 1]; end
    for index=order
        o.correct_near_wall=index==2;
        timer=tic;
        [values{index},info]=blake.velocity(s,f,x,o);
        samples(index,repetition)=toc(timer);
        phases{index,repetition}=info;
        expected=0;
        if index==2, expected=input.expected_corrected_targets; end
        assert(info.corrected_target_count==expected,'Inconsistent correction cutoff.');
    end
end
result=rmfield(input,{'sources','targets','forces'});
result.off_seconds=median(samples(1,:)); result.on_seconds=median(samples(2,:));
result.on_over_off=result.on_seconds/result.off_seconds;
result.raw_off_seconds=samples(1,:); result.raw_on_seconds=samples(2,:);
result.off_phases=phases(1,:); result.on_phases=phases(2,:);
result.accuracy=sampled_accuracy(s,f,x,values,input.wall_targets,input.close_targets);
fprintf('MATLAB: %s, corrected=%d, off=%.4fs, on=%.4fs, ratio=%.3f\n', ...
    input.case,input.expected_corrected_targets,result.off_seconds, ...
    result.on_seconds,result.on_over_off);
end

function accuracy = sampled_accuracy(s, f, x, values, wall_count, close_count)
% Check representative targets against all sources outside the timed section.
groups={1:wall_count,wall_count+1:close_count,close_count+1:size(x,2)};
names={'wall','near_wall','bulk'}; accuracy=struct();
for k=1:3
    indices=groups{k};
    if isempty(indices), continue; end
    indices=indices(round(linspace(1,numel(indices),min(32,numel(indices)))));
    [reference,info]=blake.velocity(s,f,x(:,indices), ...
        struct('method','blake','diagnostics',true));
    accuracy.(names{k})=struct('reference_target_count',numel(indices), ...
        'reference_norm',norm(reference,'fro'), ...
        'off',blake.metrics(values{1}(:,indices),reference,info.constituent_scale), ...
        'on',blake.metrics(values{2}(:,indices),reference,info.constituent_scale));
end
if wall_count>0
    accuracy.wall_residual=struct('off',max(vecnorm(values{1}(:,1:wall_count))), ...
        'on',max(vecnorm(values{2}(:,1:wall_count))));
end
end
