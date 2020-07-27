function [out, solution] = Reginska_fcn(A, b, alpha, xi)

Atb=Atransp_times_vec(A, b);
T = A'*A;
solution = inverses(T, Atb, alpha);
nx=norm(solution);
rx=norm(A*solution-b);
out=nx*((rx)^(xi));

end

function inv = inverses(T, b, alpha)
    inv=(T+alpha^2*eye(size(T)))\b;
end