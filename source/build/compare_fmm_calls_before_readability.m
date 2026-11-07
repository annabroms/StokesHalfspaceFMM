function report = compare_fmm_calls(input)
%COMPARE_FMM_CALLS Compare the existing two calls with one folded Fortran call.
% Velocity/pressure accuracy, raw wall velocity, reciprocity, then velocity timings.
%
% TWO-CALL (existing default): updated STFMM3D on physical and image sources,
% then a scalar Laplace charge/dipole FMM on images for the PN wall correction.
% MATLAB combines the two fields. The Stokes call uses four charge potentials.
% ONE-CALL (optional): local Fortran folds the same PN charges/dipoles into
% four Stokes potentials, calls LFMM3D once, then reconstructs the full field.
% Charge/dipole kernels act on all four densities and both source sets, even
% where dipole entries are zero. Saving the tree structure is beneficial, 
% but this can  add target work.
%
% Both reuse the same wall shift and horizontal recentering. Updated STFMM3D
% recenters its Stokes potentials; the new wrapper recenters the folded ones
% and preserves the original wall height in their reconstruction. Both use
% potentials and gradients only for velocity/pressure: no Laplace Hessians.
% Correctness includes a tall target and a large horizontal translation.
% Both methods remain recentered; All timing uses MATLAB TIC/TOC,
if nargin==0, input=struct(); end
c=struct('sizes',[1024 4096 16384 65536],'ratios',[0.25 1 4], ...
    'tol',1e-8,'repetitions',5,'seed',481516,'save',false);
for name=fieldnames(input).'
    assert(isfield(c,name{1}),'Unknown comparison option: %s',name{1});
    c.(name{1})=input.(name{1});
end
validateattributes(c.sizes,{'double'},{'positive','integer','vector','finite'});
validateattributes(c.ratios,{'double'},{'positive','vector','finite'});
validateattributes(c.tol,{'double'},{'scalar','>',0,'<',1,'finite'});
validateattributes(c.repetitions,{'double'},{'positive','integer','scalar','finite'});
validateattributes(c.save,{'logical'},{'scalar'});
metadata=blake.internal.fmm.manifest();
assert(~isempty(metadata.onecall_mex_path),'Build the native backend with blake.build_onecall.');
report=struct('configuration',c,'metadata',metadata);
backends={'two-call','one-call'};
fprintf(['\nImplementations compared (same half-space representation):\n' ...
    '  two-call: updated Stokes FMM on sources+images; scalar Laplace PN correction on images.\n' ...
    '            MATLAB assembles the complete field. This remains the default.\n' ...
    '  one-call: Fortran folds the PN correction into four Laplace charge/dipole potentials.\n' ...
    '            One FMM tree; Fortran reconstructs the complete field.\n' ...
    '  Both recenter coordinates and use potentials+gradients for velocity/pressure, no Hessians.\n' ...
    '  Extra dipole work can offset the saved tree, especially with many targets.\n' ...
    '  All timings are measured in MATLAB; native-call times include packing/reconstruction.\n']);

%% Accuracy against independent Blake fields, including the computed wall residual
% A tall target makes the internal vertical center nonzero. The correction
% to the fourth potential is therefore required even though the wall is z=0.
labels={'bulk and wall','shallow sources','nonzero wall and viscosity','large horizontal translation'};
heights=[1,1e-3,0.25,0.5]; walls=[0,0,1.5,1.5]; mu=[1,1,2.5,2.5];
rows=cell(8,1); row=0;
for fixture=1:4
    rng(c.seed+fixture); ns=1200; nt=37; nwall=7;
    s=round(4096*rand(3,ns))/4096; s(3,:)=heights(fixture)*(0.5+s(3,:));
    x=round(4096*rand(3,nt))/4096;
    x(3,:)=heights(fixture)*(0.2+x(3,:)); x(3,1:nwall)=0; x(3,end)=4;
    offset=[0;0;walls(fixture)];
    if fixture==4, offset(1:2)=[2^30;-2^30]; end
    s=s+offset; x=x+offset; f=randn(3,ns);
    o=struct('wall_height',walls(fixture),'mu',mu(fixture),'tol',c.tol, ...
        'requested',{{'pressure'}},'diagnostics',true,'method','blake');
    [reference,ri]=blake.fields(s,f,x,o);
    o.method='fmm'; o.diagnostics=false;
    values=cell(1,2); infos=cell(1,2);
    for method=1:2
        o.fmm_backend=backends{method};
        [values{method},infos{method}]=blake.fields(s,f,x,o);
    end
    for method=1:2
        a=values{method}; info=infos{method};
        e=blake.metrics(a.velocity,reference.velocity,ri.constituent_scale);
        wall_absolute=max(vecnorm(a.velocity(:,1:nwall)));
        wall_normalized=max(vecnorm(a.velocity(:,1:nwall))./ri.constituent_scale(1:nwall));
        difference=blake.metrics(a.velocity,values{1}.velocity,ri.constituent_scale);
        pressure_error=norm(a.pressure-reference.pressure)/norm(reference.pressure);
        row=row+1;
        rows{row}=struct('fixture',labels{fixture},'backend',backends{method}, ...
            'velocity_relative',e.global_relative,'velocity_normalized',e.constituent_normalized, ...
            'pressure_relative',pressure_error,'difference_normalized',difference.constituent_normalized, ...
            'wall_absolute',wall_absolute,'wall_normalized',wall_normalized, ...
            'fmm_calls',info.fmm_call_count);
        assert(e.constituent_normalized<100*c.tol+1e-12,'Velocity accuracy failed.');
        assert(wall_normalized<100*c.tol+1e-12,'No-slip wall check failed.');
        assert(pressure_error<100*c.tol+1e-12,'Pressure accuracy failed.');
        assert(norm(a.pressure-values{1}.pressure)/norm(reference.pressure)<100*c.tol+1e-12, ...
            'Pressure implementations disagree.');
        assert(difference.constituent_normalized<100*c.tol+1e-12,'Implementations disagree.');
        if method==2
            assert(info.fmm_call_count==1 && info.laplace_source_flag==0 && info.laplace_target_flag==2);
        end
    end
end
report.accuracy=struct2table(vertcat(rows{:}));
fprintf('\nVelocity, pressure and raw wall residuals against independent Blake evaluation:\n');
disp(report.accuracy)

%% Reciprocity: distributed force-work identity, plus all entries of a pair tensor
rng(c.seed+100);
a=rand(3,4096); a(3,:)=a(3,:)+0.1;
b=rand(3,2048); b(3,:)=b(3,:)+0.2;
fa=randn(size(a)); fb=randn(size(b));
rows=cell(2,1);
for method=1:2
    o=struct('tol',c.tol,'fmm_backend',backends{method});
    ub=blake.velocity(a,fa,b,o); ua=blake.velocity(b,fb,a,o);
    work_ab=sum(fb(:).*ub(:)); work_ba=sum(fa(:).*ua(:));
    % Norm products avoid dividing by an accidentally small signed work sum.
    scale=norm(fb(:))*norm(ub(:))+norm(fa(:))*norm(ua(:));
    defect=abs(work_ab-work_ba)/scale;
    ab=zeros(3); ba=zeros(3);
    for j=1:3
        force=zeros(3,1); force(j)=1;
        ab(:,j)=blake.velocity([0.2;0.1;0.8],force,[0.7;0.3;1.4],o);
        ba(:,j)=blake.velocity([0.7;0.3;1.4],force,[0.2;0.1;0.8],o);
    end
    pair_defect=norm(ab-ba.','fro')/norm(ab,'fro');
    rows{method}=struct('backend',backends{method},'sources_A',size(a,2),'sources_B',size(b,2), ...
        'distributed_defect',defect,'pair_defect',pair_defect);
    assert(defect<100*c.tol+1e-12 && pair_defect<100*c.tol+1e-12,'Reciprocity failed.');
end
report.reciprocity=struct2table(vertcat(rows{:}));
fprintf('\nReciprocity defects:\n'); disp(report.reciprocity)

%% Velocity scaling: warm calls, alternating order, no pressure or diagnostics
fprintf('\nVelocity timings only: complete public calls and prepared-geometry calls.\n');
% For each geometry measure both complete public calls and prepared-geometry
% calls. FMM trees are rebuilt each time. Direct checks are outside timers.
rows={}; summaries={}; case_id=0;
for ns=reshape(c.sizes,1,[])
    for ratio=reshape(c.ratios,1,[])
        case_id=case_id+1; rng(c.seed+200+case_id);
        nt=max(2,round(ns*ratio)); nwall=max(1,round(nt/10));
        s=rand(3,ns)-0.5; s(3,:)=0.1+rand(1,ns);
        x=rand(3,nt)-0.5; x(3,:)=0.1+rand(1,nt); x(3,1:nwall)=0;
        f=randn(3,ns); geom=blake.prepare(s,x);
        last_velocity=cell(1,2);
        for scope={'complete','prepared'}
            total=zeros(c.repetitions,2); kernel=total;
            for method=1:2
                o=struct('tol',c.tol,'fmm_backend',backends{method});
                if strcmp(scope{1},'complete'), blake.velocity(s,f,x,o);
                else, blake.apply(geom,f,o); end
            end
            for repeat=1:c.repetitions
                order=[1,2]; if mod(repeat,2)==0, order=[2,1]; end
                for method=order
                    o=struct('tol',c.tol,'fmm_backend',backends{method});
                    timer=tic;
                    if strcmp(scope{1},'complete')
                        [u,info]=blake.velocity(s,f,x,o);
                    else
                        [u,info]=blake.apply(geom,f,o);
                    end
                    elapsed=toc(timer);
                    total(repeat,method)=elapsed;
                    kernel(repeat,method)=info.fmm_evaluation_seconds;
                    last_velocity{method}=u;
                    rows{end+1}=struct('ns',ns,'nt',nt,'scope',scope{1}, ...
                        'backend',backends{method},'repetition',repeat, ...
                        'total_seconds',elapsed,'fmm_seconds',info.fmm_evaluation_seconds, ...
                        'fmm_calls',info.fmm_call_count); %#ok<AGROW>
                end
            end
            med=median(total,1); fmm=median(kernel,1); paired=total(:,1)./total(:,2);
            summaries{end+1}=struct('ns',ns,'nt',nt,'scope',scope{1}, ...
                'two_call_s',med(1),'one_call_s',med(2),'speedup',med(1)/med(2), ...
                'paired_min',min(paired),'paired_max',max(paired), ...
                'two_call_fmm_s',fmm(1),'one_call_fmm_s',fmm(2)); %#ok<AGROW>
            fprintf('Ns=%d Nt=%d %s: two %.4gs, one %.4gs, speedup %.3gx\n', ...
                ns,nt,scope{1},med(1),med(2),med(1)/med(2));
        end
        % Stratified wall/bulk targets, summing every source in the independent reference.
        selected=unique([round(linspace(1,nwall,min(8,nwall))), ...
            nwall+round(linspace(1,nt-nwall,min(24,nt-nwall)))]);
        [ref,ri]=blake.velocity(s,f,x(:,selected),struct('method','blake','diagnostics',true));
        for method=1:2
            e=blake.metrics(last_velocity{method}(:,selected),ref,ri.constituent_scale);
            assert(e.constituent_normalized<100*c.tol+1e-12,'Large-case direct check failed.');
        end
    end
end
report.samples=struct2table(vertcat(rows{:}));
report.timings=struct2table(vertcat(summaries{:}));
fprintf('\nMedian complete-call and prepared timings (speedup > 1 favors one call):\n');
disp(report.timings)
if c.save
    root=fileparts(fileparts(mfilename('fullpath')));
    destination=fullfile(root,'results'); if ~isfolder(destination), mkdir(destination); end
    report.output_stem=[tempname(destination) '_fmm_calls'];
    save([report.output_stem '.mat'],'report');
    writetable(report.accuracy,[report.output_stem '_accuracy.csv']);
    writetable(report.reciprocity,[report.output_stem '_reciprocity.csv']);
    writetable(report.timings,[report.output_stem '_timings.csv']);
    writetable(report.samples,[report.output_stem '_samples.csv']);
end
end
