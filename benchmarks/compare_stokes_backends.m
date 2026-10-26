function comparison = compare_stokes_backends(input)
%COMPARE_STOKES_BACKENDS Opt-in free-space velocity timing AND accuracy test.
% No wall, reflected sources, harmonic correction, or automatic test execution.
if nargin==0, input=struct(); end
c=struct('sizes',[512 8096],'ratios',[0.25 1 4],'tol',1e-6, ...
    'repetitions',7,'seed',481516,'full_pair_limit',2e7, ...
    'subset_size',256,'block_size',[256 4048],'save',true);
names=fieldnames(input);
for k=1:numel(names)
    assert(isfield(c,names{k}),'Unknown comparison option: %s',names{k});
    c.(names{k})=input.(names{k});
end
validateattributes(c.sizes,{'double'},{'integer','positive','vector','finite'});
validateattributes(c.ratios,{'double'},{'positive','finite','vector'});
validateattributes(c.tol,{'double'},{'real','finite','scalar','>',0,'<',1});
validateattributes(c.repetitions,{'double'},{'integer','positive','scalar'});
validateattributes(c.seed,{'double'},{'integer','nonnegative','scalar','finite'});
validateattributes(c.full_pair_limit,{'double'},{'positive','finite','scalar'});
validateattributes(c.subset_size,{'double'},{'integer','positive','scalar'});
validateattributes(c.block_size,{'double'},{'integer','positive','numel',2});
validateattributes(c.save,{'logical'},{'scalar'});
c.sizes=reshape(c.sizes,1,[]); c.ratios=reshape(c.ratios,1,[]);
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab'));
comparison=struct('configuration',c,'timing_schema_version',2, ...
    'problem','free-space Stokeslet velocity, mu=1', ...
    'metadata',blake.internal.fmm.manifest());
backends={'upstream','laplace-velocity'};
rows={}; differences={}; all_samples={}; case_id=0;
for ns=c.sizes
    for ratio=c.ratios
        case_id=case_id+1; nt=max(1,round(ns*ratio));
        seed=c.seed+case_id; rng(seed);
        sources=rand(3,ns)-0.5; targets=rand(3,nt)-0.5; forces=randn(3,ns);
        % All input and coincidence checks precede timed evaluations.
        assert(all(isfinite([sources(:);targets(:);forces(:)])));
        assert(isempty(intersect(sources.',targets.','rows')), ...
            'blake:SingularInteraction','A generated source coincides with a target.');
        full=double(ns)*nt<=c.full_pair_limit;
        selected=1:nt;
        if ~full
            [~,order]=sortrows(targets.');
            selected=reshape(order(unique(round(linspace(1,nt,min(c.subset_size,nt))))),1,[]);
        end
        reference=blake.internal.direct.stokes(sources,forces,targets(:,selected), ...
            c.block_size,1,true);
        velocities=cell(1,2); errors=cell(1,2);
        % Warm both backends on the exact case, then check accuracy outside timers.
        for k=1:2
            [velocities{k},~]=run_backend(backends{k},sources,forces,targets,c.tol);
            errors{k}=blake.metrics(velocities{k}(:,selected),reference.velocity,reference.scale);
            check_accuracy(errors{k},c.tol);
        end
        difference=blake.metrics(velocities{2}(:,selected),velocities{1}(:,selected),reference.scale);
        repetitions=cell(2*c.repetitions,1); index=0;
        for repeat=1:c.repetitions
            execution_order=[1 2];
            if mod(repeat,2)==0, execution_order=[2 1]; end
            for position=1:2
                k=execution_order(position);
                [velocity,timing]=run_backend(backends{k},sources,forces,targets,c.tol);
                sample=struct('case_id',case_id,'repetition',repeat,'execution_position',position, ...
                    'ns',ns,'nt',nt,'stokes_backend',backends{k}, ...
                    'backend_call_seconds',timing.backend_call_seconds, ...
                    'packing_seconds',timing.packing_seconds,'assembly_seconds',timing.assembly_seconds, ...
                    'total_seconds',timing.total_seconds, ...
                    'overhead_seconds',timing.total_seconds-timing.backend_call_seconds);
                index=index+1; repetitions{index}=sample;
                % Validate each returned velocity after its timed invocation.
                error=blake.metrics(velocity(:,selected),reference.velocity,reference.scale);
                check_accuracy(error,c.tol);
            end
        end
        samples=struct2table(vertcat(repetitions{:})); all_samples{end+1}=samples; %#ok<AGROW>
        summaries=cell(1,2);
        for k=1:2
            chosen=strcmp(samples.stokes_backend,backends{k});
            summaries{k}=blake.internal.summarize_timings(samples(chosen,:));
            row=struct('case_id',case_id,'ns',ns,'nt',nt,'seed',seed, ...
                'requested_tolerance',c.tol,'repetitions',c.repetitions, ...
                'stokes_backend',backends{k},'full_direct',full, ...
                'reference_target_count',numel(selected),'max_absolute_error',errors{k}.max_absolute, ...
                'global_relative_error',errors{k}.global_relative, ...
                'constituent_error',errors{k}.constituent_normalized);
            timing_names=fieldnames(summaries{k});
            for j=1:numel(timing_names), row.(timing_names{j})=summaries{k}.(timing_names{j}); end
            rows{end+1}=row; %#ok<AGROW>
        end
        a=summaries{1}; b=summaries{2};
        differences{end+1}=struct('case_id',case_id,'ns',ns,'nt',nt, ...
            'call_seconds_saved',a.backend_call_seconds-b.backend_call_seconds, ...
            'total_seconds_saved',a.total_seconds-b.total_seconds, ...
            'call_time_ratio',a.backend_call_seconds/b.backend_call_seconds, ...
            'total_time_ratio',a.total_seconds/b.total_seconds, ...
            'backend_max_absolute_difference',difference.max_absolute, ...
            'backend_global_relative_difference',difference.global_relative, ...
            'backend_constituent_difference',difference.constituent_normalized); %#ok<AGROW>
        fprintf('Free-space Ns=%d Nt=%d: upstream %.4gs, laplace-velocity %.4gs, total ratio %.3g\n', ...
            ns,nt,a.total_seconds,b.total_seconds,a.total_seconds/b.total_seconds);
    end
end
comparison.summary=struct2table(vertcat(rows{:}));
comparison.differences=struct2table(vertcat(differences{:}));
comparison.samples=vertcat(all_samples{:});
if c.save
    destination=fullfile(root,'results'); if ~isfolder(destination), mkdir(destination); end
    stem=tempname(destination); [~,name]=fileparts(stem);
    comparison.output_stem=fullfile(destination,['stokes_backends_' name]);
    save([comparison.output_stem '.mat'],'comparison');
    writetable(comparison.summary,[comparison.output_stem '_summary.csv']);
    writetable(comparison.differences,[comparison.output_stem '_differences.csv']);
    writetable(comparison.samples,[comparison.output_stem '_samples.csv']);
end
disp(comparison.summary); disp(comparison.differences);
end

function [velocity,timing]=run_backend(backend,sources,forces,targets,tolerance)
% Isolate one free-space evaluation so packing/reconstruction belong to its total.
if strcmp(backend,'laplace-velocity')
    [velocity,timing]=blake.internal.fmm.stokes_velocity(sources,forces,targets,tolerance);
else
    total_timer=tic; timer=tic;
    src=struct('sources',sources,'stoklet',forces);
    timing.packing_seconds=toc(timer);
    timer=tic; value=stfmm3d(tolerance,src,0,targets,1);
    timing.backend_call_seconds=toc(timer);
    timer=tic; velocity=reshape(value.pottarg,3,size(targets,2));
    assert(all(isfinite(velocity(:))),'blake:FMMFailure','Nonfinite upstream output.');
    timing.assembly_seconds=toc(timer);
    timing.total_seconds=toc(total_timer);
end
end

function check_accuracy(error,tolerance)
% Accept accuracy using measured error; never require a timing speedup.
assert(error.constituent_normalized<=100*tolerance+1e-12, ...
    'blake:AccuracyFailure','Constituent-normalized Stokes velocity error is too large.');
if ~isnan(error.global_relative)
    assert(error.global_relative<=100*tolerance+1e-11, ...
        'blake:AccuracyFailure','Global relative Stokes velocity error is too large.');
end
end
