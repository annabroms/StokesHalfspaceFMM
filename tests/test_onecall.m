function tests = test_onecall
%TEST_ONECALL Numerical acceptance of the optional single-FMM Fortran backend.
tests=functiontests(localfunctions);
end

function testBulkWallAndPressure(test)
% Include multipole work and targets above the sources to exercise vertical recentering.
require_onecall(test); blake.internal.fmm.onecall(); rng(632);
s=rand(3,1200)-0.5; s(3,:)=s(3,:)+1;
x=rand(3,37)-0.5; x(3,:)=[zeros(1,7),0.1+rand(1,29),4];
f=randn(3,1200);
o=struct('tol',1e-10,'requested',{{'pressure'}},'diagnostics',true);
o.method='blake'; [ref,ri]=blake.fields(s,f,x,o);
o.method='fmm'; [old,~]=blake.fields(s,f,x,o);
o.fmm_backend='one-call'; [a,info]=blake.fields(s,f,x,o);
e=blake.metrics(a.velocity,ref.velocity,ri.constituent_scale);
verifyLessThan(test,e.constituent_normalized,1e-8);
verifyLessThan(test,norm(a.velocity-old.velocity,'fro')/norm(ref.velocity,'fro'),1e-8);
verifyLessThan(test,norm(a.pressure-ref.pressure)/norm(ref.pressure),1e-8);
verifyLessThan(test,norm(a.pressure-old.pressure)/norm(ref.pressure),1e-8);
verifyLessThan(test,max(vecnorm(a.velocity(:,1:7))./ri.constituent_scale(1:7)),1e-8);
verifyEqual(test,info.fmm_call_count,1);
verifyEqual(test,[info.laplace_source_flag,info.laplace_target_flag],[0,2]);
verifyTrue(test,info.stokes_error_flag_exposed);
verifyEqual(test,info.fmm_evaluation_seconds,info.stokes_seconds+info.laplace_seconds);
verifyGreaterThanOrEqual(test,info.overhead_seconds,0);
end

function testGeometryUnitsAndPreparation(test)
% Shift the wall and horizontal origin; preserve physical viscosity scaling and guard reuse.
require_onecall(test); rng(31); ns=73; nt=19;
s=round(4096*rand(3,ns))/4096; s(3,:)=s(3,:)+0.5;
x=round(4096*rand(3,nt))/4096; x(3,1:4)=0;
f=randn(3,ns); offset=[2^30;-2^30;1.5];
s=s+offset; x=x+offset;
o=struct('wall_height',1.5,'mu',2.5,'tol',1e-11,'fmm_backend','one-call');
[u,~]=blake.velocity(s,f,x,o);
g=blake.prepare(s,x,1.5); [v,info]=blake.apply(g,f,o);
verifyEqual(test,u,v);
verifyTrue(test,info.guard_reused);
o.method='blake'; ref=blake.velocity(s,f,x,o);
verifyLessThan(test,norm(u-ref,'fro')/norm(ref,'fro'),1e-9);
end

function testReciprocity(test)
% Exchanging disjoint source/target clouds must preserve the force-work pairing.
require_onecall(test); rng(54);
a=rand(3,4096); a(3,:)=a(3,:)+0.1;
b=rand(3,2048); b(3,:)=b(3,:)+0.2;
fa=randn(size(a)); fb=randn(size(b));
o=struct('tol',1e-10,'fmm_backend','one-call');
ub=blake.velocity(a,fa,b,o); ua=blake.velocity(b,fb,a,o);
work_ab=sum(fb(:).*ub(:)); work_ba=sum(fa(:).*ua(:));
scale=norm(fb(:))*norm(ub(:))+norm(fa(:))*norm(ua(:));
verifyLessThan(test,abs(work_ab-work_ba)/scale,1e-8);
% A full 3-by-3 pair tensor checks each Cartesian force direction separately.
a=[0.2;0.1;0.8]; b=[0.7;0.3;1.4]; ab=zeros(3); ba=zeros(3);
for j=1:3
    force=zeros(3,1); force(j)=1;
    ab(:,j)=blake.velocity(a,force,b,o);
    ba(:,j)=blake.velocity(b,force,a,o);
end
verifyLessThan(test,norm(ab-ba.','fro')/norm(ab,'fro'),1e-9);
end

function testBoundaryCasesAndUnsupportedFields(test)
% Existing empty/coincidence conventions apply; gradients never silently request Hessians.
require_onecall(test); o=struct('fmm_backend','one-call');
[u,info]=blake.velocity(zeros(3,0),zeros(3,0),[0;0;0],o);
verifyEqual(test,u,zeros(3,1)); verifyEqual(test,info.fmm_call_count,0);
verifySize(test,blake.velocity([0;0;1],ones(3,1),zeros(3,0),o),[3 0]);
verifyError(test,@() blake.velocity([0;0;1],ones(3,1),[0;0;1],o),'blake:SingularInteraction');
o.requested={'gradient'};
verifyError(test,@() blake.fields([0;0;1],ones(3,1),[0.2;0.1;0.5],o),'blake:UnsupportedBackend');
end

function require_onecall(test)
% Missing optional compiled dependencies are incomplete checks, never passes.
assumeTrue(test,exist('fmm3d','file')==3 && ...
    exist(which('blake.internal.fmm.blake_onecall_mex'),'file')==3, ...
    'Run blake.setup and blake.build_onecall before the full acceptance suite.');
end
