function PSF = psfNSGauss(dim, sigma1, sigma2, rho)
%
%        PSF = psfGauss(dim, sigma);
%
%  This function constructs a gaussian blur PSF. 
%
%  Input: 
%    dim  -  desired dimension of the pointspread function
%            e.g., PSF = psfGauss([60,60]) creates a 60-by-60 
%            Gaussian point spread function.
%
%  Optional input parameters:
%    sigma  -  variance of the gaussian
%              Default is sigma = 2.0.
%

if ( nargin == 1 )
  sigma1 = 2; sigma2 = 2; rho = 0;
elseif (nargin == 2)
  sigma2 = sigma1; rho = 0;
elseif (nargin == 3)
  rho = 0;
end

l = length(dim);

switch l
case 1
  x = -fix(dim(1)/2):ceil(dim(1)/2)-1;
  y = 0;
case 2
  x = -fix(dim(1)/2):ceil(dim(1)/2)-1;
  y = -fix(dim(2)/2):ceil(dim(2)/2)-1;
case 3
  x = -fix(dim(1)/2):ceil(dim(1)/2)-1;
  y = -fix(dim(2)/2):ceil(dim(2)/2)-1;
otherwise
  error('illegal PSF dimension')
end

z = 0;
[X,Y] = meshgrid(x,y,z);

% invM = 1/(sigma1^2*sigma2^2 - rho^4)*[sigma2^2 -rho^2; -rho^2 sigma1^2];
% RightV = [x; y]; % RightV = [X; Y]; % 
% LeftV = RightV';
% PSF = exp( -1/2*(LeftV*(invM*RightV)) );

% PSF = zeros(dim);
% for i = 1:dim(1)
%     for j = 1:dim(2)
%         temp = [x(i); y(j)];
%         PSF(i,j) = exp( -1/2*(temp'*(invM*temp)) );
%     end
% end
PSF = exp( -1/(2*(sigma1^2*sigma2^2 - rho^4))*( sigma2^2*X.^2 - 2*rho^2*X.*Y + sigma1^2*Y.^2 ) );