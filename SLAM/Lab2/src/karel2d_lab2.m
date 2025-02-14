%-------------------------------------------------------
% University of Zaragoza
% Authors:  J. Neira, J. Tardos
%-------------------------------------------------------
% slam 2D, explore data association algorithms
%-------------------------------------------------------
clear all;
close all;
randn('state', 0);
rand('state', 0);
addpath 'tools';

% determines execution and display modes
global configuration;

configuration.ellipses = 1;
configuration.samples = 0;
configuration.tags = 0;
configuration.odometry = 1;
configuration.noise = 1;
configuration.alpha = 0.99; % only useful is chi2inv is available
configuration.step_by_step = 0;
configuration.people = 1;
configuration.ekf_iterations = 4;
configuration.maintenance = 1;

% figure numbers
configuration.ground = 1;
configuration.map = 2;
configuration.observations = 3;
configuration.compatibility = 4;
configuration.ground_hypothesis = 5;
configuration.hypothesis = 6;
configuration.tables = 7;
configuration.correlation = 8;
configuration.covisibility = 9;

%-------------------------------------------------------
global map;
%-------------------------------------------------------
%             n: number of features
%             x: state vector
%             P: covariance matrix
%     ground_id: true ids of features mapped
%          hits: times a feature has been detected
%         first: first time a feature was detected
%     estimated: history of estimated robot location
%      odometry: history of  robot odometry
%  covisibility: matrix of features that have been covisible.
     

%-------------------------------------------------------
global ground;
%-------------------------------------------------------
%              n: number of features
%         points: location of features
%     trajectory: true robot trajectory
%         motion: true robot motion

%-------------------------------------------------------
global sensor;
%-------------------------------------------------------
sensor.range = 3.5;
sensor.maxangle = pi/2;
sensor.minangle = -sensor.maxangle;
sensor.srho = 0.01;
sensor.stita = 0.125*pi/180;

%-------------------------------------------------------
global people;
%-------------------------------------------------------
%     gx: ground x locations
%     gy: ground x locations

%-------------------------------------------------------
global results;
%-------------------------------------------------------
%     total: total number of measurements at every step
%      true: number of true positives and negatives ar every step
%     false: number of fasle positives and negatives ar every step
    
%-------------------------------------------------------
global compatibility;
%-------------------------------------------------------
%             d2: Mahalanobis distance between measurements and features
%             ic: individual compatibility between measurements and features
%     candidates: measurements and features that have at least one ic
%             AL: number of alternatives for every measurement
%             HS: total number of alternative hypotheses

%-------------------------------------------------------
global prediction;
%-------------------------------------------------------
%             n: number of predicted features
%     ground_id: ground truth ids of predicted features
%             h: predicted locations
%           HPH: predicted covariance
%             H: z = Hx
%        ground: ground true predicted feature location 


%-------------------------------------------------------
% This only if chi2inv is available
%-------------------------------------------------------
%global chi;
%chi2 = chi2inv(configuration.alpha,1:1000);
%-------------------------------------------------------
% Otherwise, this data file contains variable chi2 = chi2inv(0.99,1:1000)
%-------------------------------------------------------
load 'data/chi2';

% generate the experiment data
[ground, people] = generate_cloister_experiment;

% start with a fresh map
[map, ground] = new_map(map, ground);

% plot ground
draw_ground(ground);
pause

if configuration.people
    people.x = []; 
    people.y = [];
end

%-------------------------------------------------------
% ok, here we go
%-------------------------------------------------------

step = 1;
observations = get_observations(ground, sensor, step);
draw_observations (observations, ground, step);

% initial features
map = add_features(map, observations, 1);

results.total = [];
results.true.positives = [];
results.true.negatives = [];
results.false.positives = [];
results.false.negatives = [];

GT = zeros(1, observations.m);
H = zeros(1, observations.m);
results = store_results (results, observations, GT, H);

% plot map
configuration.name = '';
draw_map (map, ground, step);

steps = length(ground.motion);
for step = 2 : steps,
    
    disp('--------------------------------------------------------------');
    disp(sprintf('Step: %d', step));
    
    %  EKF prediction step
    [motion, odometry, map] = compute_motion(map, step);
    
    % sense
    observations = get_observations(ground, sensor, step);
    
    %data association
    [H, GT, compatibility] = data_association(map, observations, step);
    
    % update EKF step
    map = EKF_update (map, observations, H);
    
    % only new features with no neighbours
    map = add_new_features (map, observations, step, H);
    
    % map maintenance:
    %
    % map.hits(i): number of times feature i has been observed
    % map.first(i): step in which feature i was first observed
    %
    % unreliable: features seen only once, more than two steps ago

    % unreliable = ; 
    % map = erase_features(map, unreliable);
    
    draw_map (map, ground, step);
    results = store_results(results, observations, GT, H);
end

show_results(map, ground, results);
