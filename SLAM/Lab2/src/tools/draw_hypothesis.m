function draw_hypothesis (prediction, observations, compatibility, H, name, color)
%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
global configuration chi2;

if configuration.step_by_step
    
    vehicle.x = [0 0 0]';
    vehicle.P = zeros(3, 3);
    
    figure(configuration.hypothesis); clf; axis equal; hold on;
    draw_vehicle(vehicle.x, vehicle.P, 'k');
    draw_sensor_range(vehicle.x, 'k');
    
    draw_prediction (prediction, compatibility.candidates.features);
    draw_obs(observations);
    
    % paired observations
    Es = find(H);
    [ix, iy, ind] = obs_rows(Es);
    xe = observations.z(ix);
    ye = observations.z(iy);
    
    % paired features
    Fs = H(find(H));
    [ix, iy, ind] = obs_rows(Fs);
    xf = prediction.h(ix);
    yf = prediction.h(iy);
    
    %pairings
    for p=1:length(Es),
        %set(h, 'LineWidth', 1.5);
        arrow([xe(p) ye(p)], [xf(p) yf(p)], 'r');
    end
    
    d2 = joint_mahalanobis2 (prediction, observations, H);
    dof = 2*length(find(H));
    title([name ': ' sprintf('%d ', H) sprintf(', d^2 %f, chi^2 %f', d2, chi2(dof))]);
    %analyze_hypothesis (prediction, observations, H);
    pause
end