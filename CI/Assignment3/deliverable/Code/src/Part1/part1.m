clear all;
clc;
close all;

fprintf('Starting HDR reconstruction process...\n');

% Add necessary directories to MATLAB path
fprintf('Adding required directories to path...\n');
addpath('../Part1');
addpath('../Part2');
addpath('../Part3');

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read stack of images with their exposure
directory = ('../Data/');   % change dir to run a different stack
[file_names, exposures] = parse_files(directory);

%% --- Set smoothing parameter (lambda) values ---
step = 20;

%% --- Set weights options ---
% Tent weight function
t = 1:128;
tent_weights = cat(2, t, t(end:-1:1)); 

% No weight function
no_weights = ones(256);

%%  Set smoothing parameter (lambda) values ---
lambda_full = 50;      % typical smoothing (full method)
lambda_no_smooth = 1e-6; % very low lambda to effectively disable smoothness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

fprintf('Preparing matrices from images in %s...\n', directory);
% Prepare Z, B subsampling the images 
[Z, B] = prepare_matrices_gsolve(file_names, exposures, step);
fprintf('Matrices prepared. Starting processing for 4 different configurations...\n');

% Create arrays for our test cases
weightings = {tent_weights, tent_weights, no_weights, no_weights};
lambdas = [lambda_full, lambda_no_smooth, lambda_full, lambda_no_smooth];
method_names = {'tent_smooth', 'tent_nosmooth', 'nowt_smooth', 'nowt_nosmooth'};
titles = {'Tent Weighting and Full Lambda', 'Tent Weighting and No Smoothness', ...
          'No Weighting and Full Lambda', 'No Weighting and No Smoothness'};

% Process each case and save radiance maps in a cell array
radiance_maps = cell(1, 4);

for i = 1:4
    fprintf('\n[%d/4] Processing: %s\n', i, titles{i});
    % Get parameters for this case
    weight = weightings{i};
    lambda = lambdas(i);
    method_name = method_names{i};
    
    fprintf('  Computing response curves...\n');
    % Compute response curves for each channel
    [g_red]   = gsolve(Z(:, :, 1), B, lambda, weight);
    [g_green] = gsolve(Z(:, :, 2), B, lambda, weight);
    [g_blue]  = gsolve(Z(:, :, 3), B, lambda, weight);
    
    fprintf('  Plotting and saving response curves...\n');
    % Plot and save response curves with title
    plot_g(g_red, g_green, g_blue, ['Results/g_' method_name '.png'], ['Camera Response Function: ' titles{i}]);
    
    fprintf('  Computing radiance map...\n');
    % Compute radiance map
    radiance_map = get_radiance_map(file_names, [g_red g_green g_blue], weight, exposures);
    radiance_maps{i} = radiance_map;
    
    fprintf('  Saving HDR file: Results/hdr_image_%s.hdr\n', method_name);
    % Save HDR file (exponentiate the radiance map for proper dynamic range)
    hdrwrite(exp(radiance_map), ['Results/hdr_image_' method_name '.hdr']);

end

% Create a figure with the radiance maps for each configuration with the 3 channels
fig_all = figure('Visible', 'off', 'Position', [100, 100, 800, 800]);

% Initialize variables to store the global min and max values
global_min = Inf;
global_max = -Inf;

% First pass: Calculate the global min and max values
for i = 1:4
    % Convert the radiance map to grayscale by averaging the RGB channels
    % grayscale_radiance_map = mean(exp(radiance_maps{i}), 3);
    grayscale_radiance_map = mean(radiance_maps{i}, 3);

    % Update global min and max values
    global_min = min(global_min, min(grayscale_radiance_map(:)));
    global_max = max(global_max, max(grayscale_radiance_map(:)));
end

% Second pass: Display the images with the calculated color limits
for i = 1:4
    subplot(2, 2, i);
    % scaled_radiance_map = scale_and_gamma(exp(radiance_maps{i}));
    scaled_radiance_map = radiance_maps{i};

    % Convert the radiance map to grayscale by averaging the RGB channels
    grayscale_radiance_map = mean(scaled_radiance_map, 3);
    % Display the grayscale radiance map
    imagesc(grayscale_radiance_map);
    axis image;
    title(titles{i}, 'FontSize', 11);
    colormap('jet');
    caxis([global_min, global_max]); % Set the color limits for comparison
end

% Add a single color bar for the entire figure
h = colorbar;
h.Position = [0.93 0.11 0.02 0.815];
sgtitle('Greyscale Radiance Maps for Different Configurations', 'FontSize', 14);
fprintf('  Saving radiance map figure for all configurations: Results/radiance_maps_all.png\n');
saveas(fig_all, 'Results/radiance_maps_all.png');
close(fig_all);

fprintf('HDR reconstruction process completed successfully!\n');