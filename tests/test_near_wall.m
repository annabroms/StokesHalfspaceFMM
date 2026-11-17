function tests = test_near_wall
%TEST_NEAR_WALL Check optional direct replacement and shared cutoff conventions.
tests=functiontests(localfunctions);
end

function testSelectionAndPhysicalDistance(test)
% Compare bounded searches with tiny independent distance matrices under scaling.
blake.internal.near_wall_targets;
[s,~,x]=fixture();
for scale=[1e-9 1 1e9]
    g=blake.prepare(scale*s,scale*x,0,false);
    for tolerance=[1e-6 1e-9 1e-12 1e-300]
        [selected,radius]=blake.internal.near_wall_targets(g,tolerance);
        distance=zeros(g.ns,g.nt);
        for m=1:g.nt
            distance(:,m)=vecnorm(g.sources-g.targets(:,m)).';
        end
        verifyEqual(test,selected,min(distance,[],1)/g.span<=radius);
    end
    [selected,radius]=blake.internal.near_wall_targets(g,1e-12,scale*1e-5);
    verifyEqual(test,selected,[true false false true]);
    verifyEqual(test,radius,scale*1e-5/g.span,'RelTol',1e-14);
end
end

function testAllDirectAndUnsafeGeometry(test)
% Noncoincident unresolved pairs use direct sums and never require an FMM call.
blake.internal.near_wall_apply;
[s,f,x]=fixture(); x=x(:,1:2);
o=struct('correct_near_wall',true,'tol',1e-12,'mu',1.7);
[u,info]=blake.velocity(s,f,x,o);
reference=zeros(size(x));
for m=1:size(x,2)
    for n=1:size(s,2)
        reference(:,m)=reference(:,m)+ ...
            blake.internal.direct.blake_pair(s(:,n),x(:,m),o.mu)*f(:,n);
    end
end
verifyLessThan(test,max(vecnorm(u-reference)./max(vecnorm(reference),1)),1e-10);
verifyEqual(test,info.corrected_target_count,2);
verifyEqual(test,info.fmm_call_count,0);
verifyGreaterThan(test,norm(u(:,2)),1e-9);
g=blake.prepare([0 1;0 1;1e-16 1],[0;0;0]);
verifyFalse(test,g.fmm_guard.safe);
[u,info]=blake.apply(g,ones(3,2),o);
verifyEqual(test,info.fmm_call_count,0);
o.method='blake'; reference=blake.apply(g,ones(3,2),o);
verifyEqual(test,u,reference);
o.method='fmm';
verifyError(test,@() blake.velocity(s,zeros(size(f)),s,o),'blake:SingularInteraction');
[u,info]=blake.velocity(zeros(3,0),zeros(3,0),x,o);
verifyEqual(test,u,zeros(size(x))); verifyEqual(test,info.fmm_call_count,0);
[u,info]=blake.velocity(s,f,zeros(3,0),o);
verifySize(test,u,[3 0]); verifyEqual(test,info.fmm_call_count,0);
end

function testMixedTwoCallBackends(test)
% Corrected targets match direct sums while other targets retain two FMM calls.
require_fmm(test);
[s,f,x]=fixture(); reference=blake.velocity(s,f,x,struct('method','blake'));
geometry=blake.prepare(s,x,0,false);
for backend={'auto','upstream','laplace-velocity'}
    o=struct('correct_near_wall',true,'tol',1e-12,'stokes_backend',backend{1}, ...
        'diagnostics',true);
    [u,info]=blake.velocity(s,f,x,o);
    verifyEqual(test,info.corrected_target_count,3);
    verifyEqual(test,info.fmm_call_count,2);
    verifyEqual(test,u(:,[1 2 4]),reference(:,[1 2 4]));
    verifyLessThan(test,norm(u(:,3)-reference(:,3)),1e-11);
    verifyLessThan(test,norm(u(:,1)),1e-9);
    verifyGreaterThan(test,info.near_wall_direct_seconds,0);
    verifyEqual(test,info.direct_seconds,info.near_wall_direct_seconds);
    verifyEqual(test,info.near_wall_distance,info.near_wall_relative_radius* ...
        geometry.span);
    verifySize(test,info.constituent_scale,[1 4]);
    g=blake.prepare(s,x); [prepared,pi]=blake.apply(g,f,o);
    verifyEqual(test,prepared,u);
    verifyFalse(test,pi.guard_reused);
end
end

function testOneCallBackend(test)
% The same correction preserves the optional single-call backend on safe targets.
require_fmm(test);
assumeTrue(test,exist(which('blake.internal.fmm.blake_onecall_mex'),'file')==3, ...
    'Optional one-call FMM unavailable.');
[s,f,x]=fixture();
o=struct('correct_near_wall',true,'tol',1e-12,'fmm_backend','one-call');
[u,info]=blake.velocity(s,f,x,o);
reference=blake.velocity(s,f,x,struct('method','blake'));
verifyEqual(test,info.corrected_target_count,3);
verifyEqual(test,info.fmm_call_count,1);
verifyEqual(test,u(:,[1 2 4]),reference(:,[1 2 4]));
verifyLessThan(test,norm(u(:,3)-reference(:,3)),1e-11);
o.requested={'pressure'};
a=blake.fields(s,f,x,o); o.method='blake'; b=blake.fields(s,f,x,o);
verifyEqual(test,a.pressure([1 2 4]),b.pressure([1 2 4]));
verifyLessThan(test,abs(a.pressure(3)-b.pressure(3)),1e-10);
end

function testMultipoleFields(test)
% Check real multipole work on safe targets alongside shallow-source replacements.
require_fmm(test); rng(171139);
s=[rand(2,1200);1e-6*(1+rand(1,1200))]; s(3,end)=0.8;
x=[rand(2,67);0.1+rand(1,67)]; f=randn(size(s));
x(:,1:26)=s(:,1:26); x(3,1:13)=0; x(3,14:26)=1e-10;
[reference,ri]=blake.velocity(s,f,x,struct('method','blake','diagnostics',true));
o=struct('correct_near_wall',true,'tol',1e-12);
[u,info]=blake.velocity(s,f,x,o);
verifyEqual(test,info.corrected_target_count,26);
verifyEqual(test,info.fmm_call_count,2);
verifyEqual(test,u(:,1:26),reference(:,1:26));
verifyLessThan(test,max(vecnorm(u(:,1:13))),1e-9);
e=blake.metrics(u,reference,ri.constituent_scale);
verifyLessThan(test,e.constituent_normalized,1e-11);
verifyLessThan(test,norm(u(:,27:end)-reference(:,27:end),'fro')/ ...
    norm(reference(:,27:end),'fro'),1e-8);
% An explicit smaller physical cutoff leaves the less-close source uncorrected.
[s,f,x]=fixture(); o.near_wall_distance=1e-5;
[~,info]=blake.velocity(s,f,x,o);
verifyEqual(test,info.corrected_target_count,2);
verifyEqual(test,info.near_wall_distance,1e-5,'RelTol',1e-14);
end

function testDistributedReciprocity(test)
% Exercise unequal multipole clouds with close pairs and a direct roundoff baseline.
require_fmm(test); rng(171140);
a=[rand(2,1200);1e-6*(1+rand(1,1200))];
b=[rand(2,701);0.2+rand(1,701)]; b(:,1:13)=a(:,1:13); b(3,1:13)=1e-10;
fa=randn(size(a)); fb=randn(size(b));
o=struct('correct_near_wall',true,'tol',1e-12);
ab=fb.*blake.velocity(a,fa,b,o); ba=fa.*blake.velocity(b,fb,a,o);
defect=abs(sum(ab(:))-sum(ba(:)))/(sum(abs(ab(:)))+sum(abs(ba(:))));
o.method='blake';
ab=fb.*blake.velocity(a,fa,b,o); ba=fa.*blake.velocity(b,fb,a,o);
direct_defect=abs(sum(ab(:))-sum(ba(:)))/(sum(abs(ab(:)))+sum(abs(ba(:))));
verifyLessThan(test,defect,max(1e-8,100*direct_defect));
end

function testFieldsAndTraction(test)
% Replace complete independent fields rather than mixing corrected velocities and derivatives.
require_fmm(test);
[s,f,x]=fixture();
o=struct('correct_near_wall',true,'tol',1e-12,'mu',1.7, ...
    'requested',{{'velocity','pressure','gradient','traction'}}, ...
    'normals',repmat([0;0;1],1,size(x,2)));
[a,info]=blake.fields(s,f,x,o);
o.method='blake'; b=blake.fields(s,f,x,o);
verifyEqual(test,info.corrected_target_count,3);
names={'velocity','pressure','grad_u','traction'};
for k=1:numel(names)
    va=a.(names{k}); vb=b.(names{k});
    if strcmp(names{k},'grad_u')
        verifyEqual(test,va(:,:,[1 2 4]),vb(:,:,[1 2 4]));
        va=va(:,:,3); vb=vb(:,:,3);
    else
        verifyEqual(test,va(:,[1 2 4]),vb(:,[1 2 4]));
        va=va(:,3); vb=vb(:,3);
    end
    verifyLessThan(test,norm(va(:)-vb(:))/max(norm(vb(:)),1),1e-10);
end
end

function testLinearityShiftAndReciprocity(test)
% Preserve units and reciprocal work when direct and accelerated targets coexist.
require_fmm(test);
[s,f,x]=fixture();
o=struct('correct_near_wall',true,'tol',1e-12);
u=blake.velocity(s,f,x,o);
verifyEqual(test,blake.velocity(s,2*f,x,o),2*u,'RelTol',1e-12);
shift=[7;-4;1.3]; o.wall_height=1.3; o.mu=1.7;
a=blake.velocity(s+shift,f,x+shift,o);
o.method='blake'; b=blake.velocity(s+shift,f,x+shift,o);
verifyLessThan(test,max(vecnorm(a-b)./max(vecnorm(b),1)),1e-10);
a=s(:,[1 3]); b=a; b(1,:)=b(1,:)+[1e-7 .1];
fa=[1 .3;-.4 .7;.2 -.6]; fb=[.5 -.2;.1 -.3;-.8 1];
o=struct('correct_near_wall',true,'tol',1e-12);
ab=fb.*blake.velocity(a,fa,b,o); ba=fa.*blake.velocity(b,fb,a,o);
defect=abs(sum(ab(:))-sum(ba(:)))/(sum(abs(ab(:)))+sum(abs(ba(:))));
verifyLessThan(test,defect,2e-12);
end

function testDefaultNoOpAndOptionValidation(test)
% Disabled correction preserves defaults; direct routes accept a validated no-op.
require_fmm(test);
[s,f,x]=fixture(); a=blake.velocity(s,f,x);
b=blake.velocity(s,f,x,struct('correct_near_wall',false)); verifyEqual(test,a,b);
for method={'blake','ggv-direct'}
    o=struct('method',method{1}); a=blake.velocity(s,f,x,o);
    o.correct_near_wall=true; [b,info]=blake.velocity(s,f,x,o);
    verifyEqual(test,a,b); verifyEqual(test,info.corrected_target_count,0);
end
verifyTrue(test,throws(@() blake.internal.options(struct('correct_near_wall',1))));
for distance={0,-1,Inf,[1 2],false}
    verifyTrue(test,throws(@() blake.internal.options(struct('near_wall_distance',distance{1}))));
end
end

function [s,f,x] = fixture()
% Supply an identical portable fixture for MATLAB and Python selector comparisons.
s=[0 .5 .7;0 .2 .6;1e-6 2e-5 .8];
f=[1 -.3 .7;.2 .4 -.8;-.1 .9 .5];
x=[0 .5 .2 .70000005;0 .2 .3 .6;0 1e-10 1 .8];
end

function result = throws(action)
% Detect invalid scalar options without relying on version-specific validator identifiers.
result=false;
try
    action();
catch
    result=true;
end
end

function require_fmm(test)
% Mark missing compiled dependencies incomplete rather than passing numerical checks.
assumeTrue(test,exist('fmm3d','file')==3 && exist('lfmm3d','file')==2, ...
    'FMM3D unavailable: run blake.setup.');
end
