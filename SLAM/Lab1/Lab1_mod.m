%-------------------------------------------------------------------------
% University of Zaragoza
%
% Author: J. Neira
%
% Description: SLAM for Karel the robot in 1D
%-------------------------------------------------------------------------
% Clear workspace, close all figures, and set random seed for reproducibility
clear all;
close all;
randn('state', 1);
rand('state', 1);
format long

%-------------------------------------------------------------------------
% Configuration Parameters
%-------------------------------------------------------------------------
global config;
config.step_by_step = 0;        % Toggle step-by-step visualization
config.steps_per_map = 1000;    % Number of robot motions per local map
config.fig = 0;                 % Figure counter for unique plots

%-------------------------------------------------------------------------
% World Characteristics
%-------------------------------------------------------------------------
global world;
world.true_point_locations = [0.5:1:100000]';   % True feature locations (static)
world.true_robot_location = 0;                  % Robot's true location in the world (dynamic)
  
%-------------------------------------------------------------------------
% Robot Characteristics
%-------------------------------------------------------------------------
global robot;
robot.factor_x = 0.1;  % 10% odometry error
robot.true_uk = 1;     % Ground True motion per step (1m)

%-------------------------------------------------------------------------
% Sensor Characteristics
%-------------------------------------------------------------------------
global sensor;
sensor.factor_z = 0.01;  % 1% measurement error
sensor.range_min = 0;    % Minimum range
sensor.range_max = 2;    % Maximum range

%-------------------------------------------------------------------------
% Map Details
%-------------------------------------------------------------------------
global map;

%            R0: absolute location of base reference for map
%         hat_x: estimated robot and feature locations
%         hat_P: robot and feature covariance matrix
%             n: number of features in map
%        true_x: true robot and feature location (with respect to R0)
%      true_ids: true label of features in map (according to world)
%  stats.true_x: true robotlocation wrt R0 for the whole trajectory
% stats.error_x: robotlocation error at every step of the trajectory
% stats.sigma_x: robot location uncertainty at every step of the trajectory
%  stats.cost_t: computational cost (elapsed time) for each step
%-------------------------------------------------------------------------

%-------------------------------------------------------------------------
% measurements (not global, but this is the structure)
%            z_k: measured distance to all visible features
%            R_k: covariance of z_k
%          ids_f: true ids of the ones that are already in the map
%          ids_n: true ids of the ones that new to the map
%        z_pos_f: positions in z_r of ids_f
%        z_pos_n: positions in z_r of ids_n
%        x_pos_f: positions in hat_x of ids_f
%            z_f: z_k(ids_f), measurements to features already in the map
%            R_f: R_k(ids_f), meas. cov. to features already in the map
%            z_n: z_k(ids_n), measurements to features new to the map
%            R_n: R_k(ids_n), Meas. cov. to features new to the map
%-------------------------------------------------------------------------
-------------------------------------------------------------------

%-------------------------------------------------------------------------
% Main Program
%-------------------------------------------------------------------------

[map] = Kalman_filter_slam(map, config.steps_per_map);
display_map_results(map);

%-------------------------------------------------------------------------
% Function Definitions
%-------------------------------------------------------------------------

function [map] = Kalman_filter_slam(map, steps)
    % Main SLAM loop using Kalman Filter

    global world;

    % Initialize map
    map.true_x = [0];
    map.n = 0;
    map.R0 = world.true_robot_location;
    map.hat_x = [0];
    map.hat_P = [0];
    map.true_ids = [0];
    map.stats.true_x = [0];
    map.stats.error_x = [];
    map.stats.sigma_x = [];
    map.stats.cost_t = [];

    for k = 1:steps
        tstart = tic;

        if k > 1
            [map] = compute_motion(map);
        end

        % Get measurements
        [measurements] = get_measurements_and_data_association(map);

        % Update map for known features
        if ~isempty(measurements.z_f)
            [map] = update_map(map, measurements);
        end

        % Add new features to the map
        if ~isempty(measurements.z_n)
            [map] = add_new_features(map, measurements);
        end

        % Record statistics
        map.stats.error_x = [map.stats.error_x; ...
                             (map.stats.true_x(end) - map.hat_x(1))];
        map.stats.sigma_x = [map.stats.sigma_x; sqrt(map.hat_P(1, 1))];
        map.stats.cost_t = [map.stats.cost_t; toc(tstart)];
    end
end

function [map] = compute_motion(map)
    % Simulate robot motion and update odometry
    global config;
    global robot;
    global world;

    % Update true robot location
    world.true_robot_location = world.true_robot_location + robot.true_uk;

    % Compute odometry error
    sigma_xk = robot.factor_x * robot.true_uk;
    error_k = randn(1) * sigma_xk;

    % Update estimated robot location
    map.hat_x(1) = map.hat_x(1) + robot.true_uk + error_k;
    map.hat_P(1, 1) = map.hat_P(1, 1) + sigma_xk^2;

    % Update true map location
    map.true_x(1) = map.true_x(1) + robot.true_uk;
    map.stats.true_x = [map.stats.true_x; ...
                        map.stats.true_x(end) + robot.true_uk];

    % Optional: Display step-by-step motion
    if config.step_by_step
        fprintf('Move to %.2f...\n', map.true_x(1));
        plot_correlation(map.hat_P);
        pause;
    end
end

%-------------------------------------------------------------------------
% Function: get_measurements_and_data_association
%-------------------------------------------------------------------------
% Purpose:
%   Simulate sensor measurements and perform data association.
%
% Inputs:
%   - map: Current state of the map.
%
% Outputs:
%   - measurements: Struct containing sensor measurements and associated data.
%-------------------------------------------------------------------------

function [measurements] = get_measurements_and_data_association(map)

    global world;
    global sensor;

    % Compute distances to all points in the world
    distances = world.true_point_locations - world.true_robot_location;

    % Identify visible points based on sensor range
    visible_ids = find((distances >= sensor.range_min) & (distances <= sensor.range_max));
    visible_d = distances(visible_ids);
    n_visible = length(visible_ids);

    % Add measurement noise (sensor error)
    sigma_z = sensor.factor_z * visible_d; % Measurement noise std. deviation
    error_z = randn(n_visible, 1) .* sigma_z; % Random Gaussian error
    measurements.z_k = visible_d + error_z; % Noisy measurements
    measurements.R_k = diag(sigma_z.^2); % Measurement covariance matrix

    % Data association: Separate known and new features
    measurements.ids_f = intersect(map.true_ids, visible_ids); % Features already in the map
    measurements.ids_n = setdiff(visible_ids, map.true_ids);   % New features

    % Map feature positions to their indices
    measurements.z_pos_f = find(ismember(visible_ids, measurements.ids_f));
    measurements.z_pos_n = find(ismember(visible_ids, measurements.ids_n));
    measurements.x_pos_f = find(ismember(map.true_ids, visible_ids));

    % Extract measurements for known features
    measurements.z_f = measurements.z_k(measurements.z_pos_f);
    measurements.R_f = measurements.R_k(measurements.z_pos_f, measurements.z_pos_f);

    % Extract measurements for new features
    measurements.z_n = measurements.z_k(measurements.z_pos_n);
    measurements.R_n = measurements.R_k(measurements.z_pos_n, measurements.z_pos_n);
end

%-------------------------------------------------------------------------
% Function: update_map
%-------------------------------------------------------------------------
% Purpose:
%   Update the map with measurements of existing features using the Kalman Filter.
%
% Inputs:
%   - map: Current map state.
%   - measurements: Sensor measurements and data associations.
%
% Outputs:
%   - map: Updated map state.
%-------------------------------------------------------------------------

function [map] = update_map(map, measurements)

    global config;

    % TODO: Implement the Kalman Filter update equations
    %  - Compute H_k: Measurement model Jacobian
    %  - Compute y_k: Innovation vector (difference between expected and actual measurements)
    %  - Compute S_k: Innovation covariance
    %  - Compute K_k: Kalman gain
    %  - Update map.hat_x and map.hat_P

    if config.step_by_step
        fprintf('Update map with %d features...\n', length(measurements.ids_f));
        plot_correlation(map.hat_P);
        pause;
    end
end

%-------------------------------------------------------------------------
% Function: add_new_features
%-------------------------------------------------------------------------
% Purpose:
%   Add new features detected by the sensor to the map.
%
% Inputs:
%   - map: Current map state.
%   - measurements: Sensor measurements and data associations.
%
% Outputs:
%   - map: Updated map state with new features added.
%-------------------------------------------------------------------------

function [map] = add_new_features(map, measurements)

    global config;
    global world;

    % TODO: Implement the logic to add new features
    %  - Update map.hat_x: Add new feature positions
    %  - Update map.hat_P: Expand covariance matrix
    %  - Update map.true_ids: Add new feature IDs
    %  - Update map.true_x: Add ground truth positions
    %  - Increment map.n: Number of features in the map

    if config.step_by_step
        fprintf('Add %d new features to the map...\n', length(measurements.ids_n));
        plot_correlation(map.hat_P);
        pause;
    end
end

%-------------------------------------------------------------------------
% Function: display_map_results
%-------------------------------------------------------------------------
% Purpose:
%   Display results of the SLAM process, including estimation error, 
%   uncertainty bounds, and computational cost.
%
% Inputs:
%   - map: Final map state after SLAM.
%-------------------------------------------------------------------------

function display_map_results(map)

    global config;

    % Plot map estimation error with uncertainty bounds
    config.fig = config.fig + 1;
    figure(config.fig);
    axis([0 length(map.hat_x) -2*max(sqrt(diag(map.hat_P))) 2*max(sqrt(diag(map.hat_P)))]);
    grid on; hold on;
    plot(map.true_ids, map.hat_x - map.true_x, 'ro', 'LineWidth', 2);
    plot(map.true_ids, 2*sqrt(diag(map.hat_P)), 'b+', 'LineWidth', 2);
    plot(map.true_ids, -2*sqrt(diag(map.hat_P)), 'b+', 'LineWidth', 2);
    xlabel('Feature number (robot = 0)');
    ylabel('meters (m)');
    title('Map estimation error + 2σ bounds');

    % Plot correlation matrix
    config.fig = config.fig + 1;
    figure(config.fig);
    plot_correlation(map.hat_P);
    title(sprintf('Correlation matrix (size: %d)', size(map.hat_P, 1)));

    % Plot cost per step
    config.fig = config.fig + 1;
    figure(config.fig);
    grid on; hold on;
    plot(map.stats.true_x, map.stats.cost_t, 'r-', 'LineWidth', 2);
    xlabel('Step');
    ylabel('Seconds');
    title('Cost per step');

    % Plot cumulative computational cost
    config.fig = config.fig + 1;
    figure(config.fig);
    grid on; hold on;
    plot(map.stats.true_x, cumsum(map.stats.cost_t), 'r-', 'LineWidth', 2);
    xlabel('Step');
    ylabel('Seconds');
    title('Cumulative cost');

    % Plot robot estimation error with uncertainty bounds
    config.fig = config.fig + 1;
    figure(config.fig);
    axis([0 map.stats.true_x(end) -2*max(map.stats.sigma_x) 2*max(map.stats.sigma_x)]);
    grid on; hold on;
    plot(map.stats.true_x, map.stats.error_x, 'r-', 'LineWidth', 2);
    plot(map.stats.true_x, 2*map.stats.sigma_x, 'b-', 'LineWidth', 2);
    plot(map.stats.true_x, -2*map.stats.sigma_x, 'b-', 'LineWidth', 2);
    xlabel('Meters (m)');
    ylabel('Meters (m)');
    title('Robot estimation error + 2σ bounds');
end

%-------------------------------------------------------------------------
% Function: plot_correlation
%-------------------------------------------------------------------------
% Purpose:
%   Visualize the correlation matrix.
%
% Inputs:
%   - P: Covariance matrix.
%-------------------------------------------------------------------------

function plot_correlation(P)

    ncol = 256; % Number of colors in the colormap

    % Define colormap (hot-to-cold)
    cmap = hsv2rgb([linspace(2/3, 0, ncol)' 0.9*ones(ncol,1) ones(ncol,1)]);
    cmap(1, :) = [0 0 0]; % Black for zero correlation
    colormap(cmap);

    % Compute and plot the correlation matrix
    corr = correlation(P);
    imagesc(abs(corr), [0 1]); % Show absolute values
    axis image;
    colorbar;
end

%-------------------------------------------------------------------------
% Function: correlation
%-------------------------------------------------------------------------
% Purpose:
%   Compute the correlation matrix from the covariance matrix.
%
% Inputs:
%   - Cov: Covariance matrix.
%
% Outputs:
%   - Corr: Correlation matrix.
%-------------------------------------------------------------------------

function Corr = correlation(Cov)

    sigmas = sqrt(diag(Cov));
    Corr = diag(1 ./ sigmas) * Cov * diag(1 ./ sigmas);
end

