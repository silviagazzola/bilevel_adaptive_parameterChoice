function [alpha1,output] = newtonstepReginska_sqrt(Cb, Chb, alpha, nb, Atbn, lambda)

original_alpha = alpha;
alpha = sqrt(alpha); 
[f,f1_aux,f2_aux] = function_and_derivative(Cb, Chb, alpha, nb, Atbn, lambda);

f1 = (f1_aux)/(2*alpha);
f2 = (alpha*f2_aux-f1_aux)/(4*alpha*original_alpha);

alpha1=original_alpha-f1/f2;

% 
% % Newton's method on the log
% 
% alpha1=alpha-(f*f1)/(f*f2-f1^2);

output.f=f; %log(f); %f; 
output.f1=f1; %f1/f;%f1; 
output.f2=f2; %(f*f2-f1^2)/f^2;%f2; 
end

% Extra functions
    function [inv1,inv2,inv3,inv4] = inverses(T,alpha)
        e1=zeros(size(T,1),1); e1(1)=1;
        inv1=(T+alpha^2*eye(size(T)))\e1;
        inv2=(T+alpha^2*eye(size(T)))\inv1;
        inv3=(T+alpha^2*eye(size(T)))\inv2;
        inv4=(T+alpha^2*eye(size(T)))\inv3;
        inv1=inv1(1); inv2=inv2(1); inv3=inv3(1); inv4=inv4(1); 
    end

    function [f,f1,f2] = function_and_derivative(Cb, Chb, alpha, nb, Atbn, lambda)
        T = Cb*Cb';
        [~, inv2, inv3, inv4] = inverses(T, alpha);
        [funa,funa1,funa2] = g(inv2,inv3,inv4,alpha); 

        T = Chb*Chb';
        [~, inv2, inv3, inv4] = inverses(T, alpha);
        [funb_aux,funb1_aux,funb2_aux] = g(inv2,inv3,inv4,alpha); 
        funb = funb_aux^(lambda);
        funb1 = lambda*funb_aux^(lambda-1)*funb1_aux;
        funb2 = lambda*(lambda-1)*funb_aux^(lambda-2)*funb1_aux+lambda*funb_aux^(lambda-1)*funb2_aux;
       
%         f = alpha^2*Atbn*nb*funa*funb;
%         f1 = Atbn*nb*(2*alpha*funa*funb+alpha^2*funa1*funb+alpha^2*funa*funb1);
%         f2 = Atbn*nb*(2*funa*funb+4*alpha*funa1*funb+4*alpha*funa*funb1+2*alpha^2*funa1*funb1+alpha^2*funa*funb2+alpha^2*funa2*funb);

        f = alpha^2*Atbn^(lambda)*nb*funa*funb;
        f1 = Atbn^(lambda)*nb*(2*alpha*funa*funb+alpha^2*funa1*funb+alpha^2*funa*funb1);
        f2 = Atbn^(lambda)*nb*(2*funa*funb+4*alpha*funa1*funb+4*alpha*funa*funb1+2*alpha^2*funa1*funb1+alpha^2*funa*funb2+alpha^2*funa2*funb);

      
% f = alpha^2*Atbn*nb*funa*funb;
% f1 = Atbn*nb*(2*lambda*alpha^(2*lambda-1)*funa*funb+alpha^(2*lambda-1)*funa1*funb+alpha^(2*lambda-1)*funa*funb1);
% f2 = Atbn*nb*(2*lambda*(2*lambda-1)*alpha^(2*lambda-2)*funa*funb ...
%     + 2*lambda*alpha^(2*lambda-1)*funa1*funb ...
%     + 2*lambda*alpha^(2*lambda-1)*funa1*funb1 ...
%     + (2*lambda-1)*alpha^(2*lambda-2)*funa1*funb ...
%     + (2*lambda-1)*alpha^(2*lambda-2)*funa*funb1 ...
%     + alpha^(2*lambda-1)*funa2*funb ...
%     + 2*alpha^(2*lambda-1)*funa1*funb1*alpha^(2*lambda-1)*funa1*funb1 ...
%     + alpha^(2*lambda-1)*funa*funb2);

    end

    function [out, out1, out2] = g(inv2,inv3,inv4,alpha)
        out = sqrt(inv2);
        out1 = -2*alpha*(inv2)^(-1/2)*inv3;
        %out2 = -2*(inv2)^(-1/2)*inv3+2*alpha^2*(inv2)^(-3/2)*inv3*(-) + 12 *alpha^2*(inv2)^(-1/2)*inv4;
        out2 = -2*(inv2)^(-1/2)*inv3-4*alpha^2*(inv2)^(-3/2)*inv3^2 + 12 *alpha^2*(inv2)^(-1/2)*inv4;
    end
