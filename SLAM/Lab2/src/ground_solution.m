function GT = ground_solution(map, observations),
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
%-------------------------------------------------------

GT = [];
for i = 1 : observations.m
    f = find(map.ground_id == observations.ground_id(i));
    if length(f) >= 1
        GT = [GT f(1)];
    else
        GT = [GT 0];
    end
end    

