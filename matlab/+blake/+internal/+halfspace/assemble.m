function fields = assemble(stokes, harmonic, z, mu, order)
%ASSEMBLE Subtract the complete Papkovich-Neuber field; scale velocity only.
if nargin == 0, self_test(); fields = []; return; end
u = stokes.velocity - harmonic.gradient.*z;
u(3,:) = u(3,:) + harmonic.potential;
fields = struct('velocity',u/mu);
if order >= 2
    fields.pressure = stokes.pressure - 2*harmonic.gradient(3,:);
end
if order >= 3
    grad = stokes.gradient;
    for i = 1:3
        for j = 1:3
            correction = -z.*reshape(harmonic.hessian(i,j,:),1,[]);
            if j == 3, correction = correction-harmonic.gradient(i,:); end
            if i == 3, correction = correction+harmonic.gradient(j,:); end
            grad(i,j,:) = grad(i,j,:) + reshape(correction,1,1,[]);
        end
    end
    fields.grad_u = grad/mu;
end
end

function self_test()
% Verify signs at z=0, including pressure without viscosity scaling.
s.velocity = zeros(3,1); s.pressure = 7;
h.potential = 2; h.gradient = [1;2;3];
v = blake.internal.halfspace.assemble(s,h,0,2,2);
assert(isequal(v.velocity,[0;0;1]) && v.pressure == 1);
end
