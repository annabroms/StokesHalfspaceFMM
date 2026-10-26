function out = stokes(sources, forces, targets, block_size, order, diagnostics)
%STOKES Blocked free-space Stokes fields, mu=1, including 1/(8*pi).
if nargin==0, self_test(); out=[]; return; end
nt=size(targets,2); ns=size(sources,2);
out=struct('velocity',zeros(3,nt),'scale',zeros(1,nt));
if order>=2, out.pressure=zeros(1,nt); end
if order>=3, out.gradient=zeros(3,3,nt); end
for t0=1:block_size(1):nt
    ti=t0:min(t0+block_size(1)-1,nt);
    for s0=1:block_size(2):ns
        si=s0:min(s0+block_size(2)-1,ns);
        r=cell(1,3);
        for k=1:3, r{k}=targets(k,ti).'-sources(k,si); end
        a=1./sqrt(r{1}.^2+r{2}.^2+r{3}.^2); a3=a.^3;
        f=forces(:,si);
        rf=r{1}.*f(1,:)+r{2}.*f(2,:)+r{3}.*f(3,:);
        magnitude=zeros(size(a));
        for i=1:3
            v=(f(i,:).*a+r{i}.*rf.*a3)/(8*pi);
            out.velocity(i,ti)=out.velocity(i,ti)+sum(v,2).';
            if diagnostics, magnitude=magnitude+v.^2; end
            if order<3, continue; end
            for j=1:3
                v=(-f(i,:).*r{j}.*a3+(i==j)*rf.*a3+r{i}.*f(j,:).*a3 ...
                    -3*r{i}.*rf.*r{j}.*a.^5)/(8*pi);
                out.gradient(i,j,ti)=out.gradient(i,j,ti)+reshape(sum(v,2),1,1,[]);
            end
        end
        if order>=2, out.pressure(ti)=out.pressure(ti)+sum(rf.*a3,2).'/(4*pi); end
        if diagnostics, out.scale(ti)=out.scale(ti)+sum(sqrt(magnitude),2).'; end
    end
end
end

function self_test()
% Check an axial Stokeslet against its closed-form velocity and pressure.
a=blake.internal.direct.stokes([0;0;0],[0;0;1],[0;0;2],[1 1],3,true);
assert(norm(a.velocity-[0;0;1/(8*pi)])<1e-15);
assert(abs(a.pressure-1/(16*pi))<1e-15);
assert(abs(trace(a.gradient))<1e-15);
end
