

function out= Bound_Reginska(Cb, Chb, alpha, nb, Atbn, lambda)

T = Cb*Cb';
[~, invx2] = inverses(T, alpha);
bnr=alpha^2*nb*sqrt(invx2);
T = Chb*Chb';
[~, invr2] = inverses(T, alpha);
bnx=Atbn*sqrt(invr2);

out=((bnx)^(lambda)*(bnr));
end

function [inv1,inv2] = inverses(T, alpha)
    e1=zeros(size(T,1)); e1(1)=1;    
    inv1=(T+alpha^2*eye(size(T)))\e1;
    inv2=(T+alpha^2*eye(size(T)))\inv1;
    inv1=inv1(1); inv2=inv2(1);
end