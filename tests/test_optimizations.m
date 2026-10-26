function tests = test_optimizations
%TEST_OPTIMIZATIONS Correctness-only coverage; never invoke performance drivers.
tests=functiontests(localfunctions);
end

function testTimingSummaries(test)
% Median-of-sums and median overhead must come from raw paired samples.
blake.internal.finalize_timing(); blake.internal.summarize_timings();
a=table([1;9;9],[9;1;9],[10;10;18],[11;30;20],[1;20;2], ...
    'VariableNames',{'stokes_seconds','laplace_seconds','fmm_evaluation_seconds', ...
    'total_seconds','overhead_seconds'});
s=blake.internal.summarize_timings(a);
verifyEqual(test,s.fmm_evaluation_seconds,10); verifyEqual(test,s.overhead_seconds,2);
a.overhead_seconds(1)=-1;
verifyError(test,@() blake.internal.summarize_timings(a),'blake:InvalidTiming');
end

function testPreparationAssessment(test)
% Unsafe FMM proximity is cached without prohibiting a direct evaluation.
sources=[0 1;0 1;1e-16 1]; targets=[0;0;0];
[g,p]=blake.prepare(sources,targets);
verifyFalse(test,g.fmm_guard.safe); verifyEqual(test,g.fmm_guard.version,1);
verifyGreaterThanOrEqual(test,p.guard_seconds,0);
u=blake.apply(g,ones(3,2),struct('method','blake'));
verifyTrue(test,all(isfinite(u(:))));
g=blake.prepare(sources,targets,0,false); verifyEmpty(test,g.fmm_guard);
verifyError(test,@() blake.prepare([0;0;1],[0;0;1]),'blake:SingularInteraction');
end

function testCachedAndLegacyGuards(test)
% Cached geometry is reused as strengths change; unchecked/old geometry is assessed.
require_fmm(test);
s=[0 1;0 0;1 0.7]; x=[0.2 0.4;0.3 0.1;0 1.2]; f=[1 2;3 1;2 -1];
g=blake.prepare(s,x); o=struct('method','fmm','tol',1e-10);
[a,first]=blake.apply(g,f,o); [b,second]=blake.apply(g,2*f,o);
verifyTrue(test,first.guard_reused && second.guard_reused);
verifyEqual(test,first.guard_seconds,0); verifyEqual(test,second.guard_seconds,0);
verifyLessThan(test,norm(b-2*a,'fro'),1e-12);
legacy=rmfield(g,'fmm_guard'); [c,info]=blake.apply(legacy,f,o);
verifyFalse(test,info.guard_reused); verifyLessThan(test,norm(c-a,'fro'),1e-12);
unchecked=blake.prepare(s,x,0,false); [~,info]=blake.apply(unchecked,f,o);
verifyFalse(test,info.guard_reused);
stale=g; stale.fmm_guard.version=0; [~,info]=blake.apply(stale,f,o);
verifyFalse(test,info.guard_reused);
bad=blake.prepare([0 1;0 1;1e-16 1],[0;0;0]);
for backend={'upstream','laplace-velocity'}
    o.stokes_backend=backend{1};
    verifyError(test,@() blake.apply(bad,ones(3,2),o),'blake:FMMResolution');
    verifyError(test,@() blake.apply(rmfield(bad,'fmm_guard'),ones(3,2),o),'blake:FMMResolution');
end
end

function testFreeSpaceAdapter(test)
% Independent free-space references cover arbitrary coordinates, forces and layouts.
require_fmm(test); blake.internal.fmm.stokes_velocity(); rng(361);
for ns=[1 7]
    for nt=[1 11]
        s=rand(3,ns)-0.5; x=rand(3,nt)-0.5;
        for direction=1:4
            f=zeros(3,ns);
            if direction<=3, f(direction,:)=1; else, f=randn(3,ns); end
            ref=blake.internal.direct.stokes(s,f,x,[5 3],1,true);
            [u,t]=blake.internal.fmm.stokes_velocity(s,f,x,1e-11);
            v=stfmm3d(1e-11,struct('sources',s,'stoklet',f),0,x,1);
            verifySize(test,u,[3 nt]);
            e=blake.metrics(u,ref.velocity,ref.scale);
            verifyLessThan(test,e.constituent_normalized,1e-9);
            verifyLessThan(test,norm(u-reshape(v.pottarg,3,nt),'fro'),1e-8*norm(ref.scale));
            verifyGreaterThanOrEqual(test,t.total_seconds,t.backend_call_seconds);
        end
    end
end
verifySize(test,blake.internal.fmm.stokes_velocity(zeros(3,0),zeros(3,0),ones(3,2),1e-9),[3 2]);
verifySize(test,blake.internal.fmm.stokes_velocity(ones(3,1),ones(3,1),zeros(3,0),1e-9),[3 0]);
end

function testMultipoleAndTimingIdentities(test)
% A modest nontrivial FMM fixture verifies both backends, not a speed claim.
require_fmm(test); rng(632);
s=rand(3,1200)-0.5; x=rand(3,37)-0.5; f=randn(3,1200);
ref=blake.internal.direct.stokes(s,f,x,[32 256],1,true);
u=blake.internal.fmm.stokes_velocity(s,f,x,1e-10);
upstream=stfmm3d(1e-10,struct('sources',s,'stoklet',f),0,x,1);
for v={u,reshape(upstream.pottarg,3,[])}
    e=blake.metrics(v{1},ref.velocity,ref.scale);
    verifyLessThan(test,e.constituent_normalized,1e-8+1e-12);
    verifyLessThan(test,e.global_relative,1e-8+1e-11);
end
s(3,:)=s(3,:)+1; x(3,:)=[zeros(1,7) 0.1+rand(1,30)];
[ref,ri]=blake.velocity(s,f,x,struct('method','blake','diagnostics',true));
for backend={'auto','upstream','laplace-velocity'}
    o=struct('method','fmm','tol',1e-10,'stokes_backend',backend{1});
    [u,info]=blake.velocity(s,f,x,o);
    e=blake.metrics(u,ref,ri.constituent_scale);
    verifyLessThan(test,e.constituent_normalized,1e-8+1e-12);
    verifyEqual(test,info.fmm_evaluation_seconds,info.stokes_seconds+info.laplace_seconds);
    verifyEqual(test,info.overhead_seconds,info.total_seconds-info.fmm_evaluation_seconds);
    verifyGreaterThanOrEqual(test,info.overhead_seconds,0);
    verifyGreaterThanOrEqual(test,info.total_seconds,sum([info.geometry_seconds ...
        info.guard_seconds info.strength_seconds info.assembly_seconds info.fmm_evaluation_seconds]));
    verifyFalse(test,info.guard_reused); verifyEqual(test,info.timing_schema_version,2);
    if strcmp(backend{1},'upstream')
        verifyEqual(test,info.stokes_backend,'upstream');
        verifyFalse(test,info.stokes_error_flag_exposed);
    else
        verifyEqual(test,info.stokes_backend,'laplace-velocity');
        verifyTrue(test,info.stokes_error_flag_exposed);
    end
end
end

function testDefaultBackendSelection(test)
% Public calls without a backend option select Laplace only for velocity.
require_fmm(test);
s=[0;0;1]; f=[1;2;3]; x=[0.2;0.3;0.4];
[u,info]=blake.velocity(s,f,x);
verifyEqual(test,info.stokes_backend,'laplace-velocity');
[a,info]=blake.fields(s,f,x);
verifyEqual(test,info.stokes_backend,'laplace-velocity');
verifyEqual(test,a.velocity,u);
[geom,~]=blake.prepare(s,x,0);
[prepared,info]=blake.apply(geom,f);
verifyEqual(test,info.stokes_backend,'laplace-velocity');
verifyEqual(test,prepared,u);
o=struct('requested',{{'velocity','pressure','gradient','traction'}},'normals',[0;0;1]);
[~,info]=blake.fields(s,f,x,o);
verifyEqual(test,info.stokes_backend,'upstream');
end

function testDerivativeBackendSelection(test)
% Pressure/gradients retain the upstream implementation and explicit rejection.
require_fmm(test);
s=[0;0;1]; f=[1;2;3]; x=[0.2;0.3;0];
for requested={{'pressure'},{'gradient'},{'traction'}}
    o=struct('method','fmm','requested',requested,'normals',[0;0;1]);
    [a,info]=blake.fields(s,f,x,o); verifyEqual(test,info.stokes_backend,'upstream');
    verifyGreaterThanOrEqual(test,info.overhead_seconds,0);
    o.stokes_backend='upstream'; b=blake.fields(s,f,x,o);
    verifyEqual(test,a,b);
    o.stokes_backend='laplace-velocity';
    verifyError(test,@() blake.fields(s,f,x,o),'blake:UnsupportedBackend');
end
end

function require_fmm(test)
% Missing compiled dependencies mark these numerical checks incomplete.
assumeTrue(test,exist('fmm3d','file')==3 && exist('lfmm3d','file')==2, ...
    'FMM3D unavailable: run blake.setup.');
end
