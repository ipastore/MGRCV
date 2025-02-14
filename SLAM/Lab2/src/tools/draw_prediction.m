function draw_prediction (prediction, which)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
global configuration;

%draw features
if nargin < 2
   which = 1:prediction.n;
end
x = prediction.h(1:2:end);
y = prediction.h(2:2:end);

for p = which,
    ind = 2*p-1:2*p;
    plot(x(p), y(p), ['b' '+']);
    if configuration.ellipses
        draw_ellipse (prediction.h(ind), prediction.HPH(ind, ind), 'b');
    end
    if configuration.tags
        ht = text(x(p)-0, y(p)-0.05, ['F' num2str(p)]);
        set(ht, 'Color', 'b');
    end
end

[i, j, ground_id] = find(prediction.ground_id(which));
plot(prediction.ground(1, ground_id), prediction.ground(2, ground_id),'r.');
