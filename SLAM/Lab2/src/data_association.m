%-------------------------------------------------------
function [H, GT, compatibility] = data_association(map, observations, step),
%-------------------------------------------------------
global configuration ground;

% individual compatibility
prediction = predict_observations (map);
compatibility = compute_compatibility (prediction, observations);

% ground truth
GT = ground_solution(map, observations);
disp(['GROUND  TRUTH: ' sprintf('%2d  ', GT)]);

% your algorithm here!
% 1. Try NN
% 2. Complete SINGLES and try it
% 3. Include people and try SINGLES
% 4. Complete JCBB and try it
% 5. Try JCBB without odometry
% 6. Eliminate features included in the map two steps ago, and never seen again

% H = NN (prediction, observations, compatibility);
H = SINGLES(prediction, observations, compatibility);

disp(['MY HYPOTHESIS: ' sprintf('%2d  ', H)]);
disp(['Correct (1/0)? ' sprintf('%2d  ', GT == H)]);
disp(' ');

draw_map (map, ground, step);
draw_observations (observations, ground, step);

draw_compatibility (prediction, observations, compatibility);

draw_hypothesis (prediction, observations, compatibility, H, configuration.name, 'b-');
draw_tables (compatibility, GT, H);
