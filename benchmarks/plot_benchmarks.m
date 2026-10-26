function plot_benchmarks(report, stem)
%PLOT_BENCHMARKS Standalone figures, with subset speedups deliberately absent.
data=report.scaling; fig=figure('Visible','off','Position',[100 100 1200 800]);
cleanup=onCleanup(@() close(fig));
tiledlayout(2,2); ratios=unique(data.target_source_ratio); heights=unique(data.height);
ax1=nexttile; hold(ax1,'on');
for r=ratios.'
    for h=heights.'
        a=sortrows(data(data.target_source_ratio==r & data.height==h,:),'ns');
        loglog(ax1,a.ns,a.total_seconds,'-o','DisplayName',sprintf('FMM Nt/Ns=%g h=%g',r,h));
        full=a.full_direct;
        loglog(ax1,a.ns(full),a.direct_seconds(full),'--','HandleVisibility','off');
    end
end
set(ax1,'XScale','log','YScale','log'); grid(ax1,'on'); xlabel(ax1,'Source count'); ylabel(ax1,'Median seconds');
title(ax1,'Solid: FMM; dashed: complete direct'); legend(ax1,'show','Location','northwest','FontSize',6);
ax2=nexttile; hold(ax2,'on');
for r=ratios.'
    for h=heights.'
        a=sortrows(data(data.target_source_ratio==r & data.height==h & data.full_direct,:),'ns');
        semilogx(ax2,a.ns,a.speedup,'-o');
    end
end
set(ax2,'XScale','log'); yline(ax2,1,':'); grid(ax2,'on'); xlabel(ax2,'Source count'); ylabel(ax2,'Measured direct / FMM');
title(ax2,'Complete-evaluation speedup only');
ax3=nexttile; t=report.tolerance;
loglog(ax3,t.requested_tolerance,t.global_relative_error,'-o', ...
    t.requested_tolerance,t.constituent_error,'-s', ...
    t.requested_tolerance,t.wall_constituent_error,'-^');
grid(ax3,'on'); xlabel(ax3,'Requested tolerance'); ylabel(ax3,'Measured error');
legend(ax3,{'Global relative','Constituent normalized','Wall normalized'},'Location','northwest','FontSize',8);
ax4=nexttile; loglog(ax4,data.ns,max(data.wall_constituent_error,1e-18),'o'); grid(ax4,'on');
xlabel(ax4,'Source count'); ylabel(ax4,'Normalized wall residual');
title(ax4,'Zeros displayed at 10^{-18}');
for ax=[ax1 ax2 ax3 ax4]
    axis(ax,'tight');
end
exportgraphics(fig,[stem '.png'],'Resolution',160);
exportgraphics(fig,[stem '.pdf'],'ContentType','vector');
end
