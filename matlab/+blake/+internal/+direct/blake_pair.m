function [B, P, G] = blake_pair(source, target, mu, wall_height)
%BLAKE_PAIR Independent Blake (1971), eq. (16), and analytic derivatives.
% B(i,j), P(j), G(i,j,k): velocity, pressure and dB(i,j)/dx(k).
if nargin == 0, self_test(); B=[]; P=[]; G=[]; return; end
if nargin < 3, mu=1; end
if nargin < 4, wall_height=0; end
validateattributes(mu,{'double'},{'real','finite','positive','scalar'});
if ~isequal(size(source),[3 1]) || ~isequal(size(target),[3 1])
    error('blake:InvalidGeometry','Pair coordinates must be 3-by-1.');
end
geom = blake.prepare(source,target,wall_height,false);
h = geom.heights;
r = geom.targets-geom.sources;
R = geom.targets-geom.images;
ri = 1/norm(r); Ri = 1/norm(R);
B = zeros(3); P = zeros(1,3); G = zeros(3,3,3);
for j = 1:3
    m = 1-2*(j==3);
    P(j) = (r(j)*ri^3-R(j)*Ri^3 ...
        -2*h*m*((j==3)*Ri^3-3*R(3)*R(j)*Ri^5))/(4*pi);
    for i = 1:3
        T = (i==j)*ri+r(i)*r(j)*ri^3;
        TI = (i==j)*Ri+R(i)*R(j)*Ri^3;
        D = (i==j)*Ri^3-3*R(i)*R(j)*Ri^5;
        E = -(i==3)*R(j)*Ri^3+((i==j)*R(3)+R(i)*(j==3))*Ri^3 ...
            -3*R(i)*R(3)*R(j)*Ri^5;
        B(i,j) = (T-TI+2*h^2*m*D-2*h*m*E)/(8*pi*mu);
        if nargout < 3, continue; end
        for k = 1:3
            dT = -(i==j)*r(k)*ri^3+((i==k)*r(j)+r(i)*(j==k))*ri^3 ...
                -3*r(i)*r(j)*r(k)*ri^5;
            dTI = -(i==j)*R(k)*Ri^3+((i==k)*R(j)+R(i)*(j==k))*Ri^3 ...
                -3*R(i)*R(j)*R(k)*Ri^5;
            dD = -3*((i==j)*R(k)+(i==k)*R(j)+R(i)*(j==k))*Ri^5 ...
                +15*R(i)*R(j)*R(k)*Ri^7;
            dE = -(i==3)*(j==k)*Ri^3+3*(i==3)*R(j)*R(k)*Ri^5 ...
                +((i==j)*(k==3)+(i==k)*(j==3))*Ri^3 ...
                -3*((i==j)*R(3)+R(i)*(j==3))*R(k)*Ri^5 ...
                -3*((i==k)*R(3)*R(j)+R(i)*(k==3)*R(j)+R(i)*R(3)*(j==k))*Ri^5 ...
                +15*R(i)*R(3)*R(j)*R(k)*Ri^7;
            G(i,j,k) = (dT-dTI+2*h^2*m*dD-2*h*m*dE)/(8*pi*mu);
        end
    end
end
end

function self_test()
% Check the wall and Lorentz reciprocal symmetry of the tensor.
y=[0.1;-0.2;0.7]; x=[0.4;0.6;1.2];
B=blake.internal.direct.blake_pair(y,[0.4;0.6;0]);
assert(norm(B,'fro')<1e-14);
A=blake.internal.direct.blake_pair(y,x);
C=blake.internal.direct.blake_pair(x,y);
assert(norm(A-C.','fro')<1e-14);
end
