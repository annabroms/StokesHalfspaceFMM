function tests = test_fields
%TEST_FIELDS Independent analytic derivatives and complete wall traction.
tests=functiontests(localfunctions);
end

function testDirectDerivativesAndTraction(test)
% Compare direct decompositions with the independently differentiated Blake tensor.
blake.fields(); rng(51);
s=[rand(2,4);0.3+rand(1,4)]; f=randn(3,4);
x=[rand(2,7);0 0.01 0.1+rand(1,5)]; normals=randn(3,7); normals=normals./vecnorm(normals);
o=struct('method','blake','mu',2.3,'requested',{{'traction'}},'normals',normals);
a=blake.fields(s,f,x,o); o.method='ggv-direct'; b=blake.fields(s,f,x,o);
verifyLessThan(test,norm(a.velocity-b.velocity,'fro'),1e-13);
verifyLessThan(test,norm(a.pressure-b.pressure),1e-13);
verifyLessThan(test,norm(a.grad_u(:)-b.grad_u(:)),1e-12);
verifyLessThan(test,norm(a.traction-b.traction,'fro'),1e-12);
for m=1:size(x,2)
    verifyLessThan(test,abs(trace(a.grad_u(:,:,m))),1e-12);
    verifyLessThan(test,abs(trace(b.grad_u(:,:,m))),1e-12);
end
o.mu=1; c=blake.fields(s,f,x,o);
verifyLessThan(test,norm(c.pressure-b.pressure),1e-14);
verifyLessThan(test,norm(c.traction-b.traction,'fro'),1e-13);
end

function testGradientFiniteDifferenceStepStudy(test)
% Finite differences are diagnostics, not production kernel implementations.
s=[0;0;0.7]; f=[1;-2;3]; x=[0.4;0.3;1.2];
o=struct('method','blake','requested',{{'gradient'}}); a=blake.fields(s,f,x,o);
steps=[1e-2 1e-3 1e-4]; errors=zeros(size(steps));
for k=1:numel(steps)
    G=zeros(3);
    for j=1:3
        delta=zeros(3,1); delta(j)=steps(k);
        G(:,j)=(blake.velocity(s,f,x+delta,o)-blake.velocity(s,f,x-delta,o))/(2*steps(k));
    end
    errors(k)=norm(G-a.grad_u,'fro');
end
verifyLessThan(test,errors(2),errors(1)/20);
verifyLessThan(test,errors(3),1e-7);
end

function testFMMCompleteFields(test)
% Verify FMM pressure, tensor ordering and full traction on an actual tree.
assumeTrue(test,exist('fmm3d','file')==3,'FMM3D unavailable: run blake.setup.');
rng(151); s=[rand(2,1200);0.2+rand(1,1200)]; f=randn(3,1200);
x=[rand(2,19);zeros(1,5) 0.1+rand(1,14)];
o=struct('method','blake','requested',{{'traction'}},'normals',repmat([0;0;1],1,19),'mu',1.7);
a=blake.fields(s,f,x,o); o.method='fmm'; o.tol=1e-10;
b=blake.fields(s,f,x,o);
verifyLessThan(test,norm(a.pressure-b.pressure)/norm(a.pressure),1e-8);
verifyLessThan(test,norm(a.grad_u(:)-b.grad_u(:))/norm(a.grad_u(:)),1e-8);
verifyLessThan(test,norm(a.traction-b.traction,'fro')/norm(a.traction,'fro'),1e-8);
for m=1:size(x,2)
    verifyLessThan(test,abs(trace(b.grad_u(:,:,m)))/max(norm(a.grad_u(:,:,m),'fro'),realmin),1e-8);
end
end

function testNormalsAndEmptyFields(test)
% Reject ambiguous normals and preserve empty output shapes.
o=struct('method','ggv-direct','requested',{{'traction'}},'normals',[0;0;2]);
verifyError(test,@() blake.fields([0;0;1],[1;2;3],[1;0;0],o),'blake:InvalidNormals');
o.normals=zeros(3,0);
a=blake.fields([0;0;1],[1;2;3],zeros(3,0),o);
verifySize(test,a.grad_u,[3 3 0]); verifySize(test,a.traction,[3 0]);
end
