function map2 = EKF_update (map, observations, H)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
global configuration;

map2 = map;

for i=1:configuration.ekf_iterations,
    prediction = predict_observations (map2);
    [d2k, Hk, Ck, hk, zk, Rk] = joint_mahalanobis2 (prediction, observations, H);
    Kk = map2.P * Hk' / (Hk * map2.P * Hk' + Rk);
    map2.x = map.x + Kk * (zk - hk - Hk*(map.x - map2.x));
end
map2.P = (eye(size(map2.P)) - Kk * Hk) * map2.P;;

map2.estimated(end+1).x = map2.x(1:3);
map2.estimated(end).P = map2.P(1:3,1:3);
map2.hits(H(find(H))) = map2.hits(H(find(H))) + 1;