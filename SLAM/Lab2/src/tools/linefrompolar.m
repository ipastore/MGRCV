function [x, y] = linefrompolar(loc, polar),

[x,y] = pol2cart(polar(2,:), polar(1,:));
p = [x; y];
pw = tpcomp(loc, p);
x = [loc(1); pw(1,:)];
y = [loc(2); pw(2,:)];
