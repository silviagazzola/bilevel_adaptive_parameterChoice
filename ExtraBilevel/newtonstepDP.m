 function [beta1,output] = newtonstepDP(Cb, beta, nb, nl)
%
%   Notation: || Ax -b ||^2_2+ (1/\beta) ||x||^2_2
%   If Cb comes from the GKB algorithms, this corresponds to a modified
%   Newton step for the DP 
%   
%   Output: - beta1: value of \beta after one step of Newton method
%           - output.f: value of the discrepancy functional evaluated in  \beta
%           - output.f1: value of the derivative of the discrepancy as a function 
%                   of \beta, evaluated in beta1
%

[f,f1] = function_and_derivative(Cb, beta, nb);

beta1=beta-(f-nl^2)/(f1);

output.f=f; 
output.f1=f1; 
end

% Extra functions
    function [inv1,inv2,inv3] = inverses(T,alpha)
        e1=zeros(size(T,1)); e1(1)=1;
        % Solving || Ax -b ||^2_2+\alpha^2 ||x||^2_2 
        inv1=(T+alpha*eye(size(T)))\e1;
        inv2=(T+alpha*eye(size(T)))\inv1;
        inv3=(T+alpha*eye(size(T)))\inv2;
        inv1=inv1(1); inv2=inv2(1); inv3=inv3(1);
    end

    function [f,f1] = function_and_derivative(Cb, beta, nb)
        T = Cb*Cb';
        [~,inv2,inv3] = inverses(T,1./beta);
        % Solving || Ax -b ||^2_2+\alpha^2 ||x||^2_2 
        f = (1./beta)^2*nb^2*inv2;
        f1 = -nb^2*(2*(1/beta)^3)*inv2 + 2* nb^2*(1./beta)^2*(1./beta^2)*inv3;
    end
