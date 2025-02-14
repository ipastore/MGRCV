function draw_vehicle (loc, cov, color)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
global configuration sensor;

robot_size = 0.1;

vertices = [1.5 -1 -1 1.5
       0    1 -1  0 ]*robot_size;
vertices = tpcomp(loc, vertices);
plot(vertices(1,:), vertices(2,:), color);
%plot(loc(1), loc(2), [color '.']);
%draw_reference (loc, color);
if configuration.ellipses
    draw_ellipse (loc, cov, color);
end
