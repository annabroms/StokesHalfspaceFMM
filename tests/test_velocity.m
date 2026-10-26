function tests = test_velocity
%TEST_VELOCITY Independent formulas, physical identities and FMM regression tests.
tests=functiontests(localfunctions);
end

function testComputationalSelfTests(~)
% Exercise each independently runnable computational unit.
blake.prepare(); blake.velocity(); blake.apply(); blake.metrics();
blake.internal.direct.blake_pair(); blake.internal.direct.blake_apply();
blake.internal.direct.stokes(); blake.internal.direct.laplace();
blake.internal.halfspace.strengths(); blake.internal.halfspace.assemble();
blake.internal.fmm.guard(); blake.internal.evaluate();
end

function testCartesianAndRandomForces(test)
% Independently compare all directions, heights and target regimes.
rng(2718); methods={'blake','ggv-direct'};
heights=10.^(-6:3:3);
for h=heights
    s=[0.1 -0.3;0.2 0.4;h 2*h];
    t=[0.1 0.8 -0.3 1.2;0.2 -0.4 0.4 0.7;0 1e-10 0.1 1e3];
    for direction=1:4
        if direction<=3
            f=zeros(3,2); f(direction,:)=[1 -0.7];
        else
            f=randn(3,2);
        end
        o=struct('method',methods{1},'diagnostics',true);
        [a,info]=blake.velocity(s,f,t,o); o.method=methods{2};
        b=blake.velocity(s,f,t,o);
        e=blake.metrics(b,a,info.constituent_scale);
        verifyLessThan(test,e.constituent_normalized,1e-12);
        for j=1:3
            verifyLessThan(test,abs(a(j,1))/info.constituent_scale(1),1e-12);
            verifyLessThan(test,abs(b(j,1))/info.constituent_scale(1),1e-12);
        end
    end
end
end

function testUnequalRandomPreparedAndBlocks(test)
% Check matrix-free application, duplicate sources and block-size independence.
rng(48); s=[rand(2,13);0.2+rand(1,13)]; f=randn(3,13);
s(:,13)=s(:,1); t=[rand(2,19);2*rand(1,19)];
o=struct('method','blake','block_size',[3 4]);
a=blake.velocity(s,f,t,o); o.block_size=[256 512];
b=blake.apply(blake.prepare(s,t),f,o);
verifyLessThan(test,norm(a-b,'fro'),1e-13);
c=zeros(size(a));
for m=1:size(t,2)
    for n=1:size(s,2)
        c(:,m)=c(:,m)+blake.internal.direct.blake_pair(s(:,n),t(:,m))*f(:,n);
    end
end
verifyLessThan(test,norm(a-c,'fro'),1e-13);
o.method='ggv-direct'; b=blake.velocity(s,f,t,o);
verifyLessThan(test,norm(a-b,'fro'),1e-13);
end

function testInputAndSingularityPolicy(test)
% Singular physical pairs must fail even for zero strengths.
s=[0;0;1]; f=zeros(3,1);
for method={'blake','ggv-direct','fmm'}
    o=struct('method',method{1});
    verifyError(test,@() blake.velocity(s,f,s,o),'blake:SingularInteraction');
end
verifyError(test,@() blake.prepare([0;0;0],[1;0;0]),'blake:InvalidGeometry');
verifyError(test,@() blake.prepare(s,[1;0;-1]),'blake:InvalidGeometry');
verifyError(test,@() blake.velocity(s,f,[1;0;0],struct('typo',1)),'blake:InvalidOptions');
verifyError(test,@() blake.apply(blake.prepare(s,[1;0;0]),f, ...
    struct('wall_height',1)),'blake:WallMismatch');
end

function testEmptyAndSingleton(test)
% Empty inputs work without invoking FMM; singleton outputs preserve shape.
o.method='fmm';
verifyEqual(test,blake.velocity(zeros(3,0),zeros(3,0),[1;0;0],o),zeros(3,1));
verifySize(test,blake.velocity([0;0;1],[1;2;3],zeros(3,0),o),[3 0]);
for method={'blake','ggv-direct'}
    o.method=method{1};
    verifySize(test,blake.velocity([0;0;1],[1;2;3],[1;0;0],o),[3 1]);
end
end

function testTranslationViscosityLinearity(test)
% Physical translation and viscosity laws expose layout and normalization errors.
rng(11); s=[rand(2,5);0.4+rand(1,5)]; t=[rand(2,7);rand(1,7)];
f=randn(3,5); g=randn(3,5); o.method='blake';
a=blake.velocity(s,f,t,o); b=blake.velocity(s,g,t,o);
verifyLessThan(test,norm(blake.velocity(s,2*f-3*g,t,o)-(2*a-3*b),'fro'),1e-13);
shift=[7;-4;1.3]; o.wall_height=1.3; o.mu=2.5;
c=blake.velocity(s+shift,f,t+shift,o);
verifyLessThan(test,norm(2.5*c-a,'fro'),1e-13);
end

function testReciprocityAndFreeSpace(test)
% Verify reciprocity and the free-space limit at fixed pair separation.
y=[0.2;-0.3;0.7]; x=[0.8;0.5;1.1];
a=blake.internal.direct.blake_pair(y,x);
b=blake.internal.direct.blake_pair(x,y);
verifyLessThan(test,norm(a-b.','fro'),1e-14);
r=x-y; S=(eye(3)/norm(r)+r*r.'/norm(r)^3)/(8*pi);
heights=[10 100 1000]; errors=zeros(size(heights));
for k=1:numel(heights)
    shift=[0;0;heights(k)]; B=blake.internal.direct.blake_pair(y+shift,x+shift);
    errors(k)=norm(B-S,'fro');
end
verifyTrue(test,all(diff(errors)<0));
verifyLessThan(test,errors(end)/norm(S,'fro'),1e-3);
end

function testIncompressibilityStepStudy(test)
% Use independent centered differences and check convergence before roundoff.
s=[0;0;0.7]; f=[1;-2;3]; x=[0.4;0.3;1.2];
steps=[1e-2 1e-3 1e-4]; errors=zeros(size(steps));
for method={'blake','ggv-direct'}
    o.method=method{1};
    for k=1:numel(steps)
        divergence=0;
        for j=1:3
            delta=zeros(3,1); delta(j)=steps(k);
            a=blake.velocity(s,f,x+delta,o); b=blake.velocity(s,f,x-delta,o);
            divergence=divergence+(a(j)-b(j))/(2*steps(k));
        end
        errors(k)=abs(divergence);
    end
    verifyLessThan(test,errors(2),errors(1)/20);
    verifyLessThan(test,errors(3),1e-7);
end
end

function testDimensionalScalingDirect(test)
% Scale both geometries and the wall, while forces and viscosity stay fixed.
check_scaling(test,{'blake','ggv-direct'},1e-12);
end

function testFMMNormalizationAndDerivatives(test)
% Separate charge/dipole signs, derivative ordering and nontrivial tree work.
require_fmm(test);
blake.internal.fmm.evaluate();
rng(93); s=rand(3,1200); t=1.2+rand(3,83); f=randn(3,1200);
src=struct('sources',s,'stoklet',f);
A=stfmm3d(1e-12,src,0,t,3);
B=blake.internal.direct.stokes(s,f,t,[64 128],3,false);
verifyLessThan(test,norm(A.pottarg-B.velocity,'fro')/norm(B.velocity,'fro'),1e-10);
verifyLessThan(test,norm(A.pretarg-B.pressure)/norm(B.pressure),1e-10);
G=permute(reshape(A.gradtarg,3,3,[]),[2 1 3]);
verifyLessThan(test,norm(G(:)-B.gradient(:))/norm(B.gradient(:)),1e-10);
for kind=1:3
    l=struct('sources',s,'charges',randn(1,1200),'dipoles',randn(3,1200));
    if kind==1, l.dipoles(:)=0; end
    if kind==2, l.charges(:)=0; end
    [L,ier]=lfmm3d(1e-12,l,0,t,3); verifyEqual(test,ier,0);
    H=blake.internal.direct.laplace(s,l.charges,l.dipoles,t,[64 128],3,false);
    verifyLessThan(test,norm(L.pottarg-H.potential)/norm(H.potential),1e-10);
    verifyLessThan(test,norm(L.gradtarg-H.gradient,'fro')/norm(H.gradient,'fro'),1e-10);
    packed=reshape(H.hessian,9,[]); packed=packed([1 5 9 4 7 8],:);
    verifyLessThan(test,norm(L.hesstarg-packed,'fro')/norm(packed,'fro'),1e-10);
end
end

function testFMMBulkWallAndHeights(test)
% Check raw wall components and well-conditioned bulk errors over source heights.
require_fmm(test); rng(927);
for h=10.^(-6:3:3)
    s=[rand(2,17);h*(1+rand(1,17))]; f=randn(3,17);
    x=[rand(2,23);0 1e-10 0.1+rand(1,21)];
    for direction=1:4
        if direction<=3, f(:)=0; f(direction,:)=randn(1,17); else, f=randn(3,17); end
        o=struct('method','blake','diagnostics',true);
        [ref,info]=blake.velocity(s,f,x,o); o.method='fmm'; o.tol=1e-10; o.diagnostics=false;
        u=blake.velocity(s,f,x,o);
        e=blake.metrics(u,ref,info.constituent_scale);
        verifyLessThan(test,e.constituent_normalized,100*o.tol+1e-12);
        verifyLessThan(test,max(abs(u(:,1)))/info.constituent_scale(1),100*o.tol+1e-12);
    end
end
% A larger mixed geometry tests the complete decomposition beyond tiny leaf sums.
s=[rand(2,1200);0.2+rand(1,1200)]; f=randn(3,1200);
x=[rand(2,81);zeros(1,13) 0.1+rand(1,68)];
o=struct('method','blake','diagnostics',true); [ref,info]=blake.velocity(s,f,x,o);
o=struct('method','fmm','tol',1e-9); u=blake.velocity(s,f,x,o);
e=blake.metrics(u,ref,info.constituent_scale);
verifyLessThan(test,e.constituent_normalized,100*o.tol+1e-12);
e=blake.metrics(u(:,14:end),ref(:,14:end),info.constituent_scale(14:end));
verifyLessThan(test,e.global_relative,100*o.tol+1e-11);
verifyError(test,@() blake.velocity([0 1;0 1;1e-16 1],ones(3,2),[0;0;0],o), ...
    'blake:FMMResolution');
verifySize(test,blake.velocity([0;0;1],[1;2;3],[1;0;0],o),[3 1]);
end

function testDimensionalScalingFMM(test)
% The FMM must respect dimensional scaling over eighteen decades.
require_fmm(test); check_scaling(test,{'fmm'},1e-8);
end

function check_scaling(test,methods,budget)
% Share one deterministic, nonzero-wall dimensional fixture across all methods.
rng(84); s=[rand(2,7);0.4+rand(1,7)]; t=[rand(2,11);rand(1,11)];
wall=1.3; s(3,:)=s(3,:)+wall; t(3,:)=t(3,:)+wall; f=randn(3,7);
for method=methods
    o=struct('method',method{1},'wall_height',wall,'mu',1.7,'tol',1e-10,'diagnostics',true);
    [u,info]=blake.velocity(s,f,t,o); o.diagnostics=false;
    for scale=10.^(-9:3:9)
        o.wall_height=scale*wall;
        v=blake.velocity(scale*s,f,scale*t,o);
        e=blake.metrics(scale*v,u,info.constituent_scale);
        verifyLessThan(test,e.constituent_normalized,budget);
    end
end
end

function require_fmm(test)
% Mark absent FMM as an incomplete test, never as a passing dependency check.
assumeTrue(test,exist('stfmm3d','file')==2 && exist('lfmm3d','file')==2 ...
    && exist('fmm3d','file')==3,'FMM3D unavailable: run blake.setup.');
end
