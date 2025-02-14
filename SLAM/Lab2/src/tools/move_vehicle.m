function ground = move_vehicle (ground, motion)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------

ground.trajectory(end+1).x = tcomp(ground.trajectory(end).x, motion.x);
ground.trajectory(end).P = zeros(3, 3);
