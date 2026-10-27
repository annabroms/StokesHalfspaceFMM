function [u, info] = singularities(kind, sources, strengths, targets, input)
%SINGULARITIES Bounded-block analytic wall kernels in physical velocity units.
if nargin == 0
    selftest(); u=[]; info=[]; return
end
timer=tic;
kind=validatestring(kind,{'rotlet','stresslet','potential-dipole'});
% These velocity-only kernels deliberately do not dispatch to a Stokeslet FMM.
if ~isfield(input,'method'), input.method='blake'; end
o=blake.internal.options(input);
if ~strcmp(o.method,'blake') || ~isequal(o.requested,{'velocity'})
    error('blake:UnsupportedBackend','Singularities support method=blake and velocity only.');
end
g=blake.prepare(sources,targets,o.wall_height,false);
validateattributes(strengths,{'double'},{'real','finite','nonsparse','size',[3 g.ns]});
if strcmp(kind,'stresslet')
    validateattributes(o.normals,{'double'},{'real','finite','nonsparse','size',[3 g.ns]});
    if any(abs(vecnorm(o.normals)-1)>1e-10)
        error('blake:InvalidNormals','Stresslet source normals must be unit vectors.');
    end
elseif ~isempty(o.normals)
    error('blake:InvalidNormals','Source normals apply only to stresslets.');
end
u=zeros(3,g.nt); scale=zeros(1,g.nt);
for t0=1:o.block_size(1):g.nt
    ti=t0:min(t0+o.block_size(1)-1,g.nt);
    for s0=1:o.block_size(2):g.ns
        si=s0:min(s0+o.block_size(2)-1,g.ns);
        r=cell(1,3); R=r;
        for k=1:3
            r{k}=g.targets(k,ti).'-g.sources(k,si);
            R{k}=g.targets(k,ti).'-g.images(k,si);
        end
        a=1./sqrt(r{1}.^2+r{2}.^2+r{3}.^2);
        b=1./sqrt(R{1}.^2+R{2}.^2+R{3}.^2);
        a3=a.^3; a5=a.^5; b3=b.^3; b5=b.^5; b7=b.^7;
        h=g.heights(si); z=g.targets(3,ti).'; f=strengths(:,si);
        A=cell(1,3); I=A; C=A;
        if strcmp(kind,'rotlet')
            % GGV (33)-(35), converted from oriented double forces to torque.
            q=-f; q(3,:)=-q(3,:); % axial reflected torque = -diag(1,1,-1)*tau
            d=[-q(2,:);q(1,:);zeros(1,numel(si))];
            Rd=R{1}.*d(1,:)+R{2}.*d(2,:);
            phi=2*Rd.*b3;
            for i=1:3
                j=mod(i,3)+1; k=mod(i+1,3)+1;
                A{i}=(f(j,:).*r{k}-f(k,:).*r{j}).*a3;
                I{i}=(q(j,:).*R{k}-q(k,:).*R{j}).*b3;
                C{i}=z.*2.*(d(i,:).*b3-3*R{i}.*Rd.*b5)-(i==3)*phi;
            end
        elseif strcmp(kind,'stresslet')
            % GGV (7), (26)-(27): oriented double layer, including its trace.
            n=o.normals(:,si); q=f; q(3,:)=-q(3,:);
            v=n; v(3,:)=-v(3,:);
            rf=r{1}.*f(1,:)+r{2}.*f(2,:)+r{3}.*f(3,:);
            rn=r{1}.*n(1,:)+r{2}.*n(2,:)+r{3}.*n(3,:);
            Rf=R{1}.*q(1,:)+R{2}.*q(2,:)+R{3}.*q(3,:);
            Rn=R{1}.*v(1,:)+R{2}.*v(2,:)+R{3}.*v(3,:);
            c=sum(f.*n,1);
            phi=4*(c.*R{3}.*b3+h.*(3*Rf.*Rn.*b5-c.*b3));
            for i=1:3
                A{i}=6*r{i}.*rf.*rn.*a5;
                I{i}=6*R{i}.*Rf.*Rn.*b5;
                grad=4*(c.*((i==3)*b3-3*R{3}.*R{i}.*b5) ...
                    +h.*(3*(q(i,:).*Rn+v(i,:).*Rf+c.*R{i}).*b5 ...
                    -15*Rf.*Rn.*R{i}.*b7));
                C{i}=z.*grad-(i==3)*phi;
            end
        else
            % Half the complete source Laplacian of Blake (16): D/(8*pi*mu).
            % Harmonic D has zero Laplacian; product rules retain h and h^2.
            for i=1:3
                A{i}=zeros(size(a)); I{i}=A{i}; C{i}=A{i};
                for j=1:3
                    m=1-2*(j==3);
                    D=(i==j)*b3-3*R{i}.*R{j}.*b5;
                    dD=-3*((i==j)*R{3}+(i==3)*R{j}+R{i}*(j==3)).*b5 ...
                        +15*R{i}.*R{j}.*R{3}.*b7;
                    dE=-(i==3)*(j==3)*b3+3*(i==3)*R{j}.*R{3}.*b5 ...
                        +((i==j)+(i==3)*(j==3))*b3 ...
                        -3*((i==j)*R{3}+R{i}*(j==3)).*R{3}.*b5 ...
                        -3*((i==3)*R{3}.*R{j}+R{i}.*R{j}+R{i}.*R{3}*(j==3)).*b5 ...
                        +15*R{i}.*R{3}.^2.*R{j}.*b7;
                    A{i}=A{i}+f(j,:).*((i==j)*a3-3*r{i}.*r{j}.*a5);
                    I{i}=I{i}+f(j,:).*D;
                    C{i}=C{i}-f(j,:).*m.*(2*D+2*h.*dD-2*dE);
                end
            end
        end
        magnitude=zeros(size(a));
        for i=1:3
            u(i,ti)=u(i,ti)+sum(A{i}-I{i}-C{i},2).'/(8*pi*o.mu);
            if o.diagnostics
                % Componentwise absolute constituents give a conservative nonzero scale.
                magnitude=magnitude+abs(A{i})+abs(I{i})+abs(C{i});
            end
        end
        if o.diagnostics, scale(ti)=scale(ti)+sum(magnitude,2).'/(8*pi*o.mu); end
    end
end
info=struct('method','blake','kernel',kind,'mu',o.mu,'wall_height',o.wall_height, ...
    'constituent_scale',scale,'total_seconds',toc(timer));
end

function selftest()
% Validate all three actual analytic kernels against independent references.
blake.internal.singularity_test();
end
