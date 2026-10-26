function fields = blake_apply(geom, forces, opts, order)
%BLAKE_APPLY Blocked explicit Blake contraction without pair tensor storage.
if nargin == 0, self_test(); fields=[]; return; end
u=zeros(3,geom.nt);
bs=opts.block_size;
for t0=1:bs(1):geom.nt
    ti=t0:min(t0+bs(1)-1,geom.nt);
    for s0=1:bs(2):geom.ns
        si=s0:min(s0+bs(2)-1,geom.ns);
        r=cell(1,3); R=cell(1,3);
        for k=1:3
            r{k}=geom.targets(k,ti).'-geom.sources(k,si);
            R{k}=geom.targets(k,ti).'-geom.images(k,si);
        end
        a=1./sqrt(r{1}.^2+r{2}.^2+r{3}.^2);
        b=1./sqrt(R{1}.^2+R{2}.^2+R{3}.^2);
        a3=a.^3; b3=b.^3; b5=b.^5;
        f=forces(:,si); g=f; g(3,:)=-g(3,:);
        rf=r{1}.*f(1,:)+r{2}.*f(2,:)+r{3}.*f(3,:);
        Rf=R{1}.*f(1,:)+R{2}.*f(2,:)+R{3}.*f(3,:);
        Rg=R{1}.*g(1,:)+R{2}.*g(2,:)+R{3}.*g(3,:);
        h=geom.heights(si);
        for i=1:3
            v=f(i,:).*a+r{i}.*rf.*a3-f(i,:).*b-R{i}.*Rf.*b3;
            v=v+2*h.^2.*(g(i,:).*b3-3*R{i}.*Rg.*b5);
            E=g(i,:).*R{3}.*b3+R{i}.*g(3,:).*b3-3*R{i}.*R{3}.*Rg.*b5;
            if i==3, E=E-Rg.*b3; end
            u(i,ti)=u(i,ti)+sum(v-2*h.*E,2).'/(8*pi*opts.mu);
        end
    end
end
fields=struct('velocity',u);
if order < 2, return; end
% The readable independent pair derivative is a reference for optional fields.
fields.pressure=zeros(1,geom.nt);
if order>=3, fields.grad_u=zeros(3,3,geom.nt); end
for m=1:geom.nt
    for n=1:geom.ns
        if order>=3
            [~,P,G]=blake.internal.direct.blake_pair(geom.sources(:,n),geom.targets(:,m),opts.mu);
            for k=1:3
                fields.grad_u(:,k,m)=fields.grad_u(:,k,m)+G(:,:,k)*forces(:,n);
            end
        else
            [~,P]=blake.internal.direct.blake_pair(geom.sources(:,n),geom.targets(:,m),opts.mu);
        end
        fields.pressure(m)=fields.pressure(m)+P*forces(:,n);
    end
end
end

function self_test()
% Compare the force contraction with independently evaluated tensor entries.
g=blake.prepare([0 0.2;0 0.3;1 0.7],[0.4;0.2;1.4]); f=[1 2;3 -1;2 1];
o=blake.internal.options(struct('method','blake','block_size',[1 1]));
a=blake.internal.direct.blake_apply(g,f,o,1); b=zeros(3,1);
for k=1:2
    b=b+blake.internal.direct.blake_pair(g.sources(:,k),g.targets)*f(:,k);
end
assert(norm(a.velocity-b)<1e-14);
end
