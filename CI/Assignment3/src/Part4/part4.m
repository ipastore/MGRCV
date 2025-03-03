clear all;
clc;
close all;

fprintf('Starting custom HDR image processing...\n');

% Add necessary directories to MATLAB path
fprintf('Adding required directories to path...\n');
addpath('../Part1'); % For parse_files, prepare_matrices_gsolve, gsolve, etc.
addpath('../Part2'); % For reinhard_tonemapping
addpath('../Part3'); % For durand_tonemapping, scale_and_gamma

% Set figures to be visible for this part
set(0, 'DefaultFigureVisible', 'on');

% Create Results directory if it doesn't exist
if ~exist('Results', 'dir')
    fprintf('Creating Results directory...\n');
    mkdir('Results');
end

%% PART 1: HDR RECONSTRUCTION
fprintf('\nPART 1: HDR RECONSTRUCTION\n');

% Read stack of images with their exposure
directory = ('../Data_1/');
fprintf('Reading images from %s\n', directory);
[file_names, exposures] = parse_files(directory);

% Display the images and their exposures
fprintf('Found %d images with the following exposures:\n', length(file_names));
for i = 1:length(file_names)
    [~, name, ext] = fileparts(file_names{i});
    fprintf('  %s%s: %.4f\n', name, ext, exposures(i));
end

% Set parameters for HDR reconstruction
step = 20;  % Subsample every 20th pixel

% Create tent weighting function (best from Part 1)
t = 1:128;
tent_weights = cat(2, t, t(end:-1:1)); 

% Set smoothing parameter (best from Part 1)
lambda = 50;  % Regular smoothing

% Prepare matrices from the images
fprintf('Preparing matrices for HDR reconstruction...\n');
[Z, B] = prepare_matrices_gsolve(file_names, exposures, step);

% Compute response curves for each channel
fprintf('Computing camera response functions...\n');
[g_red]   = gsolve(Z(:, :, 1), B, lambda, tent_weights);
[g_green] = gsolve(Z(:, :, 2), B, lambda, tent_weights);
[g_blue]  = gsolve(Z(:, :, 3), B, lambda, tent_weights);

% Plot and save response curves
fprintf('Plotting response curves...\n');
plot_g(g_red, g_green, g_blue, 'Results/custom_g_response.png', 'Camera Response Function: Custom Images');

% Compute radiance map
fprintf('Computing radiance map...\n');
radiance_map = get_radiance_map(file_names, [g_red g_green g_blue], tent_weights, exposures);

% Save HDR file
fprintf('Saving HDR file: Results/custom_hdr_image.hdr\n');
hdrwrite(exp(radiance_map), 'Results/custom_hdr_image.hdr');


%% PART 2: REINHARD TONE MAPPING (GLOBAL)
fprintf('\nPART 2: REINHARD TONE MAPPING\n');

% Load the HDR radiance map (using the one we just created)
hdr = exp(radiance_map);

% Parameters for Reinhard tone mapping (selected after experimenting)
key = 0.18;  % Middle-gray value
burn = 0.5;  % Burn parameter for highlights

% Apply Reinhard tone mapping
fprintf('Applying Reinhard tone mapping (key=%.2f, burn=%.1f)...\n', key, burn);
reinhard_img = reinhard_tonemapping(hdr, key, burn, 1);
imwrite(reinhard_img, 'Results/custom_reinhard.png');


%% PART 3: DURAND TONE MAPPING (LOCAL)
fprintf('\nPART 3: DURAND TONE MAPPING\n');

% Parameters for Durand tone mapping
dR = 4;  % Dynamic range compression factor

% Apply Durand tone mapping
fprintf('Applying Durand tone mapping (dR=%d)...\n', dR);
durand_img = durand_tonemapping(hdr, dR, 1, 'custom', 'Custom Image');
imwrite(durand_img, 'Results/custom_durand.png');

%% COMPARISON
fprintf('\nCREATING COMPARISON FIGURE\n');

% Create a side-by-side comparison of all methods
figure('Name', 'HDR Results Comparison', 'Position', [100, 100, 1200, 400]);

% Original radiance map
subplot(1, 3, 1);
% Create figure for the radiance map
display_radiance = mean(radiance_map,3);
imagesc(display_radiance);
axis image;
colormap('jet');
colorbar;
title('HDR Radiance Map', 'FontSize', 12);

% Reinhard tone mapping
subplot(1, 3, 2);
imshow(reinhard_img);
title(sprintf('Reinhard (key=%.2f, burn=%.1f)', key, burn), 'FontSize', 12);

% Durand tone mapping
subplot(1, 3, 3);
imshow(durand_img);
title(sprintf('Durand (dR=%d)', dR), 'FontSize', 12);

sgtitle('Custom HDR Image Processing Results', 'FontSize', 14);
saveas(gcf, 'Results/custom_comparison.png');

% Also create a simple HDR for comparison
simple_hdr = scale_and_gamma(hdr ./ (1 + hdr));
imwrite(simple_hdr, 'Results/custom_simple_hdr.png');

fprintf('\nCustom HDR image processing completed successfully!\n');
fprintf('Results saved in the Results directory.\n');
