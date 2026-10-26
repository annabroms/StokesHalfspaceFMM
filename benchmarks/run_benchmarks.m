function report = run_benchmarks(input)
%RUN_BENCHMARKS Reproducible full/direct-subset timings and accuracy sweeps.
% Profiles: quick (32:4:8192), default (32:4:131072), large (+524288,2097152).
% Counts listed above denote geometric sequences, not MATLAB colon syntax.
if nargin==0, input=struct(); end
c=struct('profile','default','sizes',[],'ratios',[0.25 1 4], ...
    'heights',[1e-5 1e-2 1],'repetitions',5,'tol',1e-9,'subset_size',256, ...
    'full_pair_limit',2e7,'seed',734,'save',true,'stokes_backend','auto');
names=fieldnames(input);
for k=1:numel(names)
    assert(isfield(c,names{k}),'Unknown benchmark option: %s',names{k});
    c.(names{k})=input.(names{k});
end
if isempty(c.sizes)
    switch c.profile
        case 'quick', c.sizes=2.^(5:2:13);
        case 'default', c.sizes=2.^(5:2:17);
        case 'large', c.sizes=2.^(5:2:21);
        otherwise, error('Unknown benchmark profile.');
    end
end
validateattributes(c.sizes,{'double'},{'integer','positive','vector'});
validateattributes(c.repetitions,{'double'},{'integer','positive','scalar'});
validateattributes(c.subset_size,{'double'},{'integer','positive','scalar'});
validateattributes(c.ratios,{'double'},{'positive','finite','vector'});
validateattributes(c.heights,{'double'},{'positive','finite','vector'});
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab'));
report.metadata=blake.internal.fmm.manifest(); report.configuration=c;
report.timing_schema_version=2;
% Warm MATLAB, both FMM interfaces and direct kernels before all timings.
rng(c.seed); s=[rand(2,128);0.2+rand(1,128)]; f=randn(3,128); x=rand(3,97);
blake.velocity(s,f,x,struct('method','fmm','tol',c.tol,'stokes_backend',c.stokes_backend));
blake.velocity(s,f,x,struct('method','blake'));
rows={}; all_samples={}; case_id=0;
for ns=c.sizes
    for ratio=c.ratios
        nt=max(2,round(ratio*ns)); nw=max(1,round(nt/8)); nb=nt-nw;
        for hi=1:numel(c.heights)
            seed=c.seed+ns+round(100*ratio)+1000000*hi; rng(seed);
            s=[rand(2,ns);c.heights(hi)*(1+rand(1,ns))]; f=randn(3,ns);
            x=[rand(2,nt);0.05+1.95*rand(1,nb) zeros(1,nw)];
            full=double(ns)*nt<=c.full_pair_limit;
            if full
                selected=1:nt;
            else
                selected=[representative(x,1:nb,c.subset_size) ...
                    representative(x,nb+(1:nw),c.subset_size)];
            end
            direct_options=struct('method','blake','diagnostics',true);
            [reference,reference_info]=blake.velocity(s,f,x(:,selected),direct_options);
            wall_selected=x(3,selected)==0;
            case_id=case_id+1; repetitions=cell(c.repetitions,1);
            options=struct('method','fmm','tol',c.tol,'stokes_backend',c.stokes_backend);
            for repeat=1:c.repetitions
                [u,info]=blake.velocity(s,f,x,options);
                sample=struct('case_id',case_id,'repetition',repeat,'ns',ns,'nt',nt, ...
                    'height',c.heights(hi),'stokes_backend',info.stokes_backend, ...
                    'geometry_seconds',info.geometry_seconds,'strength_seconds',info.strength_seconds, ...
                    'guard_seconds',info.guard_seconds,'assembly_seconds',info.assembly_seconds, ...
                    'stokes_seconds',info.stokes_seconds,'laplace_seconds',info.laplace_seconds, ...
                    'fmm_evaluation_seconds',info.fmm_evaluation_seconds, ...
                    'total_seconds',info.total_seconds,'overhead_seconds',info.overhead_seconds);
                timer=tic;
                blake.velocity(s,f,x(:,selected),struct('method','blake'));
                sample.direct_seconds=toc(timer);
                repetitions{repeat}=sample;
            end
            samples=struct2table(vertcat(repetitions{:}));
            all_samples{end+1}=samples; %#ok<AGROW>
            e=blake.metrics(u(:,selected),reference,reference_info.constituent_scale);
            w=blake.metrics(u(:,selected(wall_selected)),zeros(3,sum(wall_selected)), ...
                reference_info.constituent_scale(wall_selected));
            med=blake.internal.summarize_timings(samples);
            row=struct('ns',ns,'nt',nt,'bulk_targets',nb,'wall_targets',nw, ...
                'target_source_ratio',ratio,'height',c.heights(hi),'seed',seed, ...
                'requested_tolerance',c.tol,'repetitions',c.repetitions, ...
                'stokes_backend',info.stokes_backend,'full_direct',full, ...
                'reference_target_count',numel(selected),'max_absolute_error',e.max_absolute, ...
                'global_relative_error',e.global_relative,'constituent_error',e.constituent_normalized, ...
                'wall_max_absolute',max(vecnorm(u(:,nb+1:end))), ...
                'wall_constituent_error',w.constituent_normalized,'speedup',NaN);
            timing_names=fieldnames(med);
            for k=1:numel(timing_names), row.(timing_names{k})=med.(timing_names{k}); end
            if full, row.speedup=row.direct_seconds/row.total_seconds; end
            rows{end+1}=row; %#ok<AGROW>
            fprintf('Ns=%d Nt=%d h=%g: FMM calls %.4gs, total %.4gs, overhead %.4gs, error %.3g, wall %.3g%s\n', ...
                ns,nt,row.height,row.fmm_evaluation_seconds,row.total_seconds,row.overhead_seconds, ...
                e.global_relative,w.constituent_normalized, ...
                sprintf(' (direct targets %d)',numel(selected)));
        end
    end
end
report.scaling=struct2table(vertcat(rows{:}));
report.samples=vertcat(all_samples{:});
[report.tolerance,report.tolerance_samples]=tolerance_sweep(c);
report.blocks=block_study(c);
report.observations=observations(report.scaling);
if c.save
    destination=fullfile(root,'results'); if ~isfolder(destination), mkdir(destination); end
    stem=['benchmark_' c.profile '_' char(datetime('now','Format','yyyyMMdd_HHmmss'))];
    report.output_stem=fullfile(destination,stem);
    save([report.output_stem '.mat'],'report');
    writetable(report.scaling,[report.output_stem '_scaling.csv']);
    writetable(report.samples,[report.output_stem '_samples.csv']);
    writetable(report.tolerance,[report.output_stem '_tolerance.csv']);
    writetable(report.tolerance_samples,[report.output_stem '_tolerance_samples.csv']);
    writetable(report.blocks,[report.output_stem '_blocks.csv']);
    writetable(report.observations,[report.output_stem '_observations.csv']);
    plot_benchmarks(report,report.output_stem);
end
disp(report.observations);
end

function selected=representative(x,candidates,count)
% Stratify fixed validation targets over height and lateral position.
[~,order]=sortrows(x([3 1 2],candidates).');
ranks=unique(round(linspace(1,numel(candidates),min(count,numel(candidates)))));
selected=candidates(order(ranks)); selected=reshape(selected,1,[]);
end

function [table_out,samples_out]=tolerance_sweep(c)
% Hold geometry and strengths fixed while measuring requested versus achieved error.
rng(c.seed+900); ns=2048; nt=257;
s=[rand(2,ns);0.01*(1+rand(1,ns))]; f=randn(3,ns);
x=[rand(2,nt);0.05+rand(1,nt-33) zeros(1,33)];
[reference,reference_info]=blake.velocity(s,f,x,struct('method','blake','diagnostics',true));
rows={}; all_samples={};
for tol=[1e-3 1e-6 1e-9 1e-12]
    o=struct('method','fmm','tol',tol,'stokes_backend',c.stokes_backend);
    blake.velocity(s,f,x,o); repetitions=cell(c.repetitions,1);
    for k=1:c.repetitions
        [u,info]=blake.velocity(s,f,x,o);
        repetitions{k}=struct('requested_tolerance',tol,'repetition',k,'ns',ns,'nt',nt, ...
            'stokes_backend',info.stokes_backend,'total_seconds',info.total_seconds, ...
            'stokes_seconds',info.stokes_seconds,'laplace_seconds',info.laplace_seconds, ...
            'fmm_evaluation_seconds',info.fmm_evaluation_seconds,'overhead_seconds',info.overhead_seconds);
    end
    samples=struct2table(vertcat(repetitions{:})); all_samples{end+1}=samples; %#ok<AGROW>
    med=blake.internal.summarize_timings(samples);
    e=blake.metrics(u,reference,reference_info.constituent_scale);
    w=blake.metrics(u(:,end-32:end),zeros(3,33),reference_info.constituent_scale(end-32:end));
    row=struct('requested_tolerance',tol,'ns',ns,'nt',nt, ...
        'stokes_backend',info.stokes_backend,'repetitions',c.repetitions, ...
        'max_absolute_error',e.max_absolute,'global_relative_error',e.global_relative, ...
        'constituent_error',e.constituent_normalized,'wall_constituent_error',w.constituent_normalized);
    names=fieldnames(med);
    for k=1:numel(names), row.(names{k})=med.(names{k}); end
    rows{end+1}=row; %#ok<AGROW>
end
 table_out=struct2table(vertcat(rows{:})); samples_out=vertcat(all_samples{:});
end

function table_out=block_study(c)
% Include a source-loop/vectorized-target alternative instead of assuming full vectorization wins.
rng(c.seed+901); s=[rand(2,1024);0.2+rand(1,1024)]; f=randn(3,1024); x=rand(3,1024);
blocks=[64 64;128 256;256 512;512 256;1024 1]; rows={};
reference=blake.velocity(s,f,x,struct('method','blake'));
for k=1:size(blocks,1)
    o=struct('method','blake','block_size',blocks(k,:)); blake.velocity(s,f,x,o);
    times=zeros(c.repetitions,1);
    for j=1:c.repetitions
        timer=tic; u=blake.velocity(s,f,x,o); times(j)=toc(timer);
    end
    rows{end+1}=struct('target_block',blocks(k,1),'source_block',blocks(k,2), ...
        'total_seconds',median(times),'repetitions',c.repetitions, ...
        'max_absolute_difference',max(vecnorm(u-reference))); %#ok<AGROW>
end
table_out=struct2table(vertcat(rows{:}));
end

function table_out=observations(data)
% Report only measured full-evaluation crossover and descriptive log-log slopes.
ratios=unique(data.target_source_ratio); heights=unique(data.height); rows={};
for ratio=ratios.'
    for height=heights.'
        a=data(data.target_source_ratio==ratio & data.height==height,:);
        a=sortrows(a,'ns'); measured=a(a.full_direct,:);
        crossing=measured.ns(measured.speedup>=1); crossover=NaN;
        if ~isempty(crossing), crossover=min(crossing); end
        slope=NaN;
        if size(a,1)>1
            fit=polyfit(log(double(a.ns)),log(a.total_seconds),1); slope=fit(1);
        end
        rows{end+1}=struct('target_source_ratio',ratio,'source_height',height, ...
            'first_measured_crossover_ns',crossover,'observed_runtime_slope',slope); %#ok<AGROW>
    end
end
table_out=struct2table(vertcat(rows{:}));
end
