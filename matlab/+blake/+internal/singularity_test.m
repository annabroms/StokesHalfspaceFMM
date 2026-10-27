function report = singularity_test(kind)
%SINGULARITY_TEST Reproducible wall, normalization and independent FD checks.
if nargin == 0
    selftest(); report=[]; return
end
timer=tic;
state=rng; restore=onCleanup(@() rng(state));
rng(7281);
o=struct('mu',1.7,'wall_height',0.35,'diagnostics',true,'block_size',[3 2]);
s=[randn(2,4);o.wall_height+[0.08 0.3 1.2 3]];
f=randn(3,4); n=randn(3,4); n=n./vecnorm(n);
x=[s(1:2,:) randn(2,9);repmat(o.wall_height,1,13)];
wall_abs=0; wall_rel=0; power=2+strcmp(kind,'potential-dipole');
% Separate each torque/moment basis and all nine oriented stresslet products.
for count=[1 4]
    for j=1:4
        for k=1:3
            F=zeros(3,count); F(k,:)=1;
            if j==4, F=f(:,1:count); end
            if strcmp(kind,'stresslet')
                N=zeros(3,count); N(min(j,3),:)=1;
                if j==4, N=n(:,1:count); end
                o.normals=N;
            end
            [u,info]=blake.internal.singularities(kind,s(:,1:count),F,x,o);
            % A source/geometry scale stays positive even for a zero basis field.
            natural=zeros(1,size(x,2));
            for source=1:count
                natural=natural+norm(F(:,source))./vecnorm(x-s(:,source)).^power/(8*pi*o.mu);
            end
            ab=max(vecnorm(u)); re=max(vecnorm(u)./max(info.constituent_scale,natural));
            wall_abs=max(wall_abs,ab); wall_rel=max(wall_rel,re);
            assert(re<300*eps,'Wall cancellation exceeded 300 eps.');
        end
    end
end
if strcmp(kind,'stresslet'), o.normals=n; end
x=[randn(2,7);o.wall_height+0.2+rand(1,7)];
u=blake.internal.singularities(kind,s,f,x,o);
f2=randn(size(f));
w=blake.internal.singularities(kind,s,f2,x,o);
v=blake.internal.singularities(kind,s,2.3*f-0.4*f2,x,o);
assert(norm(v-(2.3*u-0.4*w),'fro')<1e-13*max(1,norm(u,'fro')));
% Translate the entire geometry and wall and vary viscosity independently.
p=o; shift=[2;-3;0.7]; p.wall_height=o.wall_height+shift(3); p.mu=2*o.mu;
v=blake.internal.singularities(kind,s+shift,f,x+shift,p);
assert(norm(2*v-u,'fro')<1e-12*max(1,norm(u,'fro')));
% Source-by-source additivity also checks blocking, unequal counts and normals.
v=zeros(size(u));
for j=1:4
    p=o; if strcmp(kind,'stresslet'), p.normals=n(:,j); end
    v=v+blake.internal.singularities(kind,s(:,j),f(:,j),x,p);
end
assert(norm(u-v,'fro')<1e-13*max(1,norm(u,'fro')));
for scale=[0.17 7]
    p=o; p.wall_height=o.wall_height*scale;
    v=blake.internal.singularities(kind,s*scale,f,x*scale,p);
    assert(norm(v*scale^power-u,'fro')<2e-12*max(1,norm(u,'fro')));
end
% Away-from-wall source differences of the existing independent Blake tensor.
y=[0.1;-0.2;1.05]; x=[0.6 -0.4;0.2 0.8;1.6 0.9];
f=[0.7;-0.5;1.1]; n=[0.3;0.4;sqrt(0.75)];
if strcmp(kind,'stresslet'), o.normals=n; end
u=blake.internal.singularities(kind,y,f,x,o);
steps=0.02*2.^(-(0:6)); errors=zeros(size(steps));
for k=1:numel(steps)
    ref=reference(kind,y,f,n,x,o,steps(k));
    errors(k)=norm(u-ref,'fro')/norm(u,'fro');
end
assert(all(errors(2:end)<errors(1:end-1)/3.5),'FD study did not converge quadratically.');
assert(errors(end)<1e-6);
% Target differences verify divergence without reusing production derivatives.
div=zeros(1,size(x,2)); step=1e-5;
for j=1:3
    d=zeros(3,1); d(j)=step;
    up=blake.internal.singularities(kind,y,f,x+d,o);
    um=blake.internal.singularities(kind,y,f,x-d,o);
    div=div+(up(j,:)-um(j,:))/(2*step);
end
div_error=norm(div)/norm(u,'fro'); assert(div_error<1e-7);
% Fixed relative separation at increasing heights verifies the free-space limit.
free=freefield(kind,x-y,f,n,o.mu); limit=zeros(1,3);
for k=1:3
    d=[0;0;10^k]; v=blake.internal.singularities(kind,y+d,f,x+d,o);
    limit(k)=norm(v-free,'fro')/norm(free,'fro');
end
assert(all(limit(2:end)<limit(1:end-1)/20)); assert(limit(end)<1e-5);
% Approach the singularity at fixed wall separation to expose sign and prefactor.
r=[0.4;-0.2;0.3]; near=zeros(1,3);
for k=1:3
    d=r*10^(-k); v=blake.internal.singularities(kind,y,f,y+d,o);
    ref=freefield(kind,d,f,n,o.mu); near(k)=norm(v-ref)/norm(ref);
end
assert(all(near(2:end)<near(1:end-1)/20)); assert(near(end)<1e-5);
% Empty sets and coincidence policy are inherited but explicitly exercised.
p=o; if strcmp(kind,'stresslet'), p.normals=zeros(3,0); end
v=blake.internal.singularities(kind,zeros(3,0),zeros(3,0),x,p); assert(~any(v(:)));
v=blake.internal.singularities(kind,y,f,zeros(3,0),o); assert(isequal(size(v),[3 0]));
caught=false;
try
    blake.internal.singularities(kind,y,zeros(3,1),y,o);
catch exception
    caught=strcmp(exception.identifier,'blake:SingularInteraction');
end
assert(caught);
report=struct('wall_absolute',wall_abs,'wall_normalized',wall_rel, ...
    'steps',steps,'reference_relative_errors',errors,'divergence_relative',div_error, ...
    'free_limit_relative',limit(end),'near_source_relative',near(end),'seconds',toc(timer));
fprintf('%s PASS: wall abs %.3e normalized %.3e; FD [%s]; div %.3e; far %.3e; near %.3e; %.3fs\n', ...
    kind,wall_abs,wall_rel,sprintf(' %.3e',errors),div_error,limit(end),near(end),report.seconds);
end

function v = reference(kind,y,f,n,x,o,step)
% Differentiate validated Blake, adding the independent flux field for stresslets.
v=zeros(size(x));
for t=1:size(x,2)
    B=blake.internal.direct.blake_pair(y,x(:,t),o.mu,o.wall_height);
    for k=1:3
        d=zeros(3,1); d(k)=step;
        plus=blake.internal.direct.blake_pair(y+d,x(:,t),o.mu,o.wall_height);
        minus=blake.internal.direct.blake_pair(y-d,x(:,t),o.mu,o.wall_height);
        if strcmp(kind,'potential-dipole')
            v(:,t)=v(:,t)+(plus-2*B+minus)*f/(2*step^2);
        else
            if strcmp(kind,'rotlet')
                e=zeros(3,1); e(k)=1; c=cross(f,e)/2;
            else
                c=f*n(k)+n*f(k);
            end
            v(:,t)=v(:,t)+(plus-minus)*c/(2*step);
        end
    end
    if strcmp(kind,'stresslet')
        % GGV (31)-(32): source flux plus image and a vertical dipole PN field.
        r=x(:,t)-y; R=r; R(3)=x(3,t)+y(3)-2*o.wall_height;
        b=1/norm(R); phi=2*R(3)*b^3;
        grad=2*([0;0;1]*b^3-3*R(3)*R*b^5);
        Q=r/norm(r)^3-R*b^3-(x(3,t)-o.wall_height)*grad+[0;0;phi];
        v(:,t)=v(:,t)+2*dot(f,n)*Q/(8*pi*o.mu);
    end
end
end

function v = freefield(kind,r,f,n,mu)
% Independent free-space formulas fix physical signs and normalization.
a=1./vecnorm(r);
if strcmp(kind,'rotlet')
    v=cross(repmat(f,1,size(r,2)),r,1).*a.^3;
elseif strcmp(kind,'stresslet')
    v=6*r.*sum(r.*f,1).*sum(r.*n,1).*a.^5;
else
    v=f.*a.^3-3*r.*sum(r.*f,1).*a.^5;
end
v=v/(8*pi*mu);
end

function selftest()
% Run each family's assertion-based validation, preserving caller random state.
for kind={'rotlet','potential-dipole','stresslet'}
    blake.internal.singularity_test(kind{1});
end
fprintf('singularity_test PASS (all families).\n');
end
