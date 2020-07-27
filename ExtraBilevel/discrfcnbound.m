function val = discrfcnbound(C, b, beta, noise)

% val = (sqrt(beta)*C + eye(size(C)))\b; %%%
val = (beta*C + eye(size(C)))\b; %%%
val = val'*val - noise^2;
