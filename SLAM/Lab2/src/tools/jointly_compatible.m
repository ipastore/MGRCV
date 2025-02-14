function [answer, d2] = jointly_compatible (prediction, observations, H)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
%-------------------------------------------------------
global configuration chi2;

d2 = joint_mahalanobis2 (prediction, observations, H);
dof = 2*length(find(H));
%dof = 2*observations.m;

answer = d2 < chi2(dof);


