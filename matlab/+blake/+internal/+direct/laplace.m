function out = laplace(sources, charges, dipoles, targets, block_size, order, diagnostics)
%LAPLACE Analytic charge/dipole potential, target gradient and Hessian.
if nargin==0, self_test(); out=[]; return; end
nt=size(targets,2); ns=size(sources,2);
out=struct('potential',zeros(1,nt),'gradient',zeros(3,nt), ...
    'potential_scale',zeros(1,nt),'gradient_scale',zeros(1,nt));
if order>=3, out.hessian=zeros(3,3,nt); end
for t0=1:block_size(1):nt
    ti=t0:min(t0+block_size(1)-1,nt);
    for s0=1:block_size(2):ns
        si=s0:min(s0+block_size(2)-1,ns);
        r=cell(1,3);
        for k=1:3, r{k}=targets(k,ti).'-sources(k,si); end
        a=1./sqrt(r{1}.^2+r{2}.^2+r{3}.^2);
        q=charges(si); d=dipoles(:,si);
        rd=r{1}.*d(1,:)+r{2}.*d(2,:)+r{3}.*d(3,:);
        phi=(q.*a+rd.*a.^3)/(4*pi);
        out.potential(ti)=out.potential(ti)+sum(phi,2).';
        magnitude=zeros(size(a));
        for i=1:3
            v=(-q.*r{i}.*a.^3+d(i,:).*a.^3-3*r{i}.*rd.*a.^5)/(4*pi);
            out.gradient(i,ti)=out.gradient(i,ti)+sum(v,2).';
            if diagnostics, magnitude=magnitude+v.^2; end
            if order<3, continue; end
            for j=1:3
                v=(q.*(3*r{i}.*r{j}.*a.^5-(i==j)*a.^3) ...
                    -3*(d(i,:).*r{j}+d(j,:).*r{i}+(i==j)*rd).*a.^5 ...
                    +15*rd.*r{i}.*r{j}.*a.^7)/(4*pi);
                out.hessian(i,j,ti)=out.hessian(i,j,ti)+reshape(sum(v,2),1,1,[]);
            end
        end
        if diagnostics
            out.potential_scale(ti)=out.potential_scale(ti)+sum(abs(phi),2).';
            out.gradient_scale(ti)=out.gradient_scale(ti)+sum(sqrt(magnitude),2).';
        end
    end
end
end

function self_test()
% Check an axial dipole and harmonic Hessian trace independently.
a=blake.internal.direct.laplace([0;0;0],0,[0;0;1],[0;0;2],[1 1],3,true);
assert(abs(a.potential-1/(16*pi))<1e-15);
assert(norm(a.gradient-[0;0;-1/(16*pi)])<1e-15);
assert(abs(trace(a.hessian))<1e-15);
end
