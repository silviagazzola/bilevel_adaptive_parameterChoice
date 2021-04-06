function discrval = discrfcn(l, A, RegM, b, nrmb, nnoise)

n = size(A,2);
if n == 1
    discrval = 0;
else
    xl = [A; l*RegM]\[b; zeros(size(RegM,1),1)];
    discrval = (norm(A*xl -b)/nrmb)^2 - nnoise^2;
end