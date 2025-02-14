%-------------------------------------------------------
function [motion, odometry, map] = compute_motion(map, step),
%-------------------------------------------------------
global ground; 

motion = ground.motion(step - 1);
ground = move_vehicle (ground, motion);
odometry = get_odometry (motion);
map = EKF_prediction (map, odometry);
