function map = erase_features (map, which)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------

[ix, iy, inde] = obs_rows(which);
inde = inde + 3;
ind = ones(size(map.x));
ind(inde) = 0;
ind = find(ind);
map.x = map.x(ind);
map.P = map.P(ind, ind);

ind = ones(map.n, 1);
ind(which) = 0;
ind = find(ind);
map.ground_id = map.ground_id(ind);
map.first = map.first(ind);
map.hits = map.hits(ind);
map.n = map.n - length(which);