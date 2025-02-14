function h = draw_ellipse(pos, cov, color)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
global chi2;

persistent CIRCLE

if isempty(CIRCLE) 
    tita = linspace(0, 2*pi,40);
    CIRCLE = [cos(tita); sin(tita)];
end

[V,D]=eig(full(cov(1:2,1:2)));
ejes=sqrt(2*chi2(2)*diag(D));
P = (V*diag(ejes))*CIRCLE;
hp = line(P(1,:)+pos(1), P(2,:)+pos(2));
set(hp,'Color', color);
set(hp, 'LineWidth', 1.5);
