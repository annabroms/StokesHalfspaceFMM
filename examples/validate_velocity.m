function report = validate_velocity(use_fmm, save_outputs)
%VALIDATE_VELOCITY Readable orientation/height/wall and dimensional-scaling checks.
if nargin<1, use_fmm=true; end
if nargin<2, save_outputs=true; end
root=fileparts(fileparts(mfilename('fullpath'))); addpath(fullfile(root,'matlab'));
methods={'blake','ggv-direct'};
if use_fmm, blake.internal.fmm.require(); methods{end+1}='fmm'; end
rng(1801); rows={}; wall=0.6;
for ns=[1 17]
    for h=10.^(-6:3:3)
        sources=[rand(2,ns);wall+h*(1+rand(1,ns))];
        targets=[rand(2,29);wall+[zeros(1,7) 1e-10*ones(1,7) logspace(-4,3,15)]];
        for direction=1:4
            forces=zeros(3,ns);
            if direction<=3, forces(direction,:)=1; else, forces=randn(3,ns); end
            o=struct('method','blake','wall_height',wall,'diagnostics',true);
            [reference,reference_info]=blake.velocity(sources,forces,targets,o);
            for method=methods
                o.method=method{1}; o.diagnostics=false; o.tol=1e-9;
                u=blake.velocity(sources,forces,targets,o);
                e=blake.metrics(u,reference,reference_info.constituent_scale);
                w=blake.metrics(u(:,1:7),zeros(3,7),reference_info.constituent_scale(1:7));
                budget=1e-12;
                if strcmp(method{1},'fmm'), budget=100*o.tol+1e-12; end
                assert(e.constituent_normalized<budget && w.constituent_normalized<budget);
                rows{end+1}=struct('method',method{1},'ns',ns,'nt',29,'source_height',h, ...
                    'force_direction',direction,'requested_tolerance',o.tol, ...
                    'max_absolute_error',e.max_absolute,'global_relative_error',e.global_relative, ...
                    'constituent_error',e.constituent_normalized,'wall_max_absolute',w.max_absolute, ...
                    'wall_constituent_error',w.constituent_normalized); %#ok<AGROW>
            end
        end
    end
end
report.validation=struct2table(vertcat(rows{:}));
rng(1802); sources=[rand(2,7);1+rand(1,7)]; targets=[rand(2,11);0.6+rand(1,11)];
forces=randn(3,7); rows={};
for method=methods
    o=struct('method',method{1},'wall_height',wall,'diagnostics',true,'tol',1e-10);
    [u,info]=blake.velocity(sources,forces,targets,o); o.diagnostics=false;
    for scale=10.^(-9:3:9)
        o.wall_height=scale*wall;
        v=blake.velocity(scale*sources,forces,scale*targets,o);
        e=blake.metrics(scale*v,u,info.constituent_scale);
        budget=1e-12; if strcmp(method{1},'fmm'), budget=1e-8; end
        assert(e.constituent_normalized<budget);
        rows{end+1}=struct('method',method{1},'coordinate_scale',scale, ...
            'rescaled_max_absolute_error',e.max_absolute,'global_relative_error',e.global_relative, ...
            'constituent_error',e.constituent_normalized); %#ok<AGROW>
    end
end
report.scaling=struct2table(vertcat(rows{:}));
disp(report.validation); disp(report.scaling);
if save_outputs
    destination=fullfile(root,'results'); if ~isfolder(destination), mkdir(destination); end
    save(fullfile(destination,'validation.mat'),'report');
    writetable(report.validation,fullfile(destination,'validation.csv'));
    writetable(report.scaling,fullfile(destination,'dimensional_scaling.csv'));
    fig=figure('Visible','off','Position',[100 100 1200 500]); cleanup=onCleanup(@() close(fig));
    tiledlayout(1,2); ax1=nexttile; hold(ax1,'on');
    for method=methods
        selected=strcmp(report.validation.method,method{1});
        loglog(ax1,report.validation.source_height(selected), ...
            max(report.validation.wall_constituent_error(selected),1e-18),'o','DisplayName',method{1});
    end
    set(ax1,'XScale','log','YScale','log'); grid(ax1,'on'); legend(ax1,'show','Location','northwest');
    xlabel(ax1,'Source height'); ylabel(ax1,'Constituent-normalized wall residual');
    title(ax1,'Zero residuals displayed at 10^{-18}');
    ax2=nexttile; hold(ax2,'on');
    for method=methods
        selected=strcmp(report.scaling.method,method{1});
        loglog(ax2,report.scaling.coordinate_scale(selected), ...
            max(report.scaling.constituent_error(selected),1e-18),'-o','DisplayName',method{1});
    end
    set(ax2,'XScale','log','YScale','log'); grid(ax2,'on'); legend(ax2,'show','Location','northwest');
    xlabel(ax2,'Coordinate scale s'); ylabel(ax2,'Normalized error in s u_s - u');
    title(ax2,'Zero errors displayed at 10^{-18}');
    axis(ax1,'tight');
    axis(ax2,'tight');
    exportgraphics(fig,fullfile(destination,'validation.png'),'Resolution',160);
    exportgraphics(fig,fullfile(destination,'validation.pdf'),'ContentType','vector');
end
end
