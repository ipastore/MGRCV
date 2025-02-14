function x = gaussian_noise(P)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
% function x = gaussian_noise(P)
%
% Generate a Gaussian noise x following N(0,P).
%-------------------------------------------------------%
[V,D] = eig(P);
x = V*(randn(size(D,1),1).*sqrt(diag(D)));