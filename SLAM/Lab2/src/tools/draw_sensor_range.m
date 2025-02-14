function draw_sensor_range (loc, color)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
global configuration sensor;

[xl, yl] = linefrompolar(loc, [sensor.range; sensor.minangle]);
plot(xl, yl, color);
[xl, yl] = linefrompolar(loc, [sensor.range; sensor.maxangle]);
plot(xl, yl, color);

tita = [sensor.minangle:1*pi/180:sensor.maxangle];
[x, y] = pol2cart(tita, ones(1,length(tita))*sensor.range);
pwa = tpcomp(loc, [x; y]);
plot(pwa(1,:), pwa(2,:), color);
