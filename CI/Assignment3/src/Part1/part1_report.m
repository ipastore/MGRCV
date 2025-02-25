clear all;
clc;
close all;

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Read stack of images with their exposure
directory = ('../Data/');   % change dir to run a different stack
[file_names, exposures] = parse_files(directory);

%% --- Set smoothing parameter (lambda) values ---
step = 20;

%% --- Set weights options---
% Tent weight function
t = 1:128;
tent_weights = cat(2, t, t(end:-1:1)); 

% No weight function
no_weights = ones(256);

%%  Set smoothing parameter (lambda) values ---
lambda_full = 50;      % typical smoothing (full method)
lambda_no_smooth = 1e-6; % very low lambda to effectively disable smoothness
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% CONFIG %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Prepare Z, B subsampling the images 
[Z, B] = prepare_matrices_gsolve(file_names, exposures, step);

% Try different weightings and lambdas
%  Full Method: with smoothing (lambda_full) and tent weighting
[g_red_full]   = gsolve(Z(:, :, 1), B, lambda_full, tent_weights);
[g_green_full] = gsolve(Z(:, :, 2), B, lambda_full, tent_weights);
[g_blue_full]  = gsolve(Z(:, :, 3), B, lambda_full, tent_weights);
plot_g(g_red_full, g_green_full, g_blue_full, 'Results/g_tent_smooth.png');
radiance_map_tent_smooth = get_radiance_map(file_names, [g_red_full g_green_full g_blue_full], tent_weights, exposures);
clf(figure(1))
figure(1)
imagesc(radiance_map_tent_smooth(:, :, 1))
axis image;
title('Radiance Map: Tent Weighting and Full Lambda');
saveas(gcf, 'Results/radiance_map_tent_smooth.png')
hdrwrite(exp(radiance_map_tent_smooth), 'Results/hdr_image_tent_smooth.hdr')

% No Smoothness: using very low lambda, but still with tent weighting 
[g_red_nosmooth]   = gsolve(Z(:, :, 1), B, lambda_no_smooth, tent_weights);
[g_green_nosmooth] = gsolve(Z(:, :, 2), B, lambda_no_smooth, tent_weights);
[g_blue_nosmooth]  = gsolve(Z(:, :, 3), B, lambda_no_smooth, tent_weights);
plot_g(g_red_nosmooth, g_green_nosmooth, g_blue_nosmooth, 'Results/g_tent_nosmooth.png');
radiance_map_tent_nosmooth = get_radiance_map(file_names, [g_red_nosmooth g_green_nosmooth g_blue_nosmooth], tent_weights, exposures);
clf(figure(2))
figure(2)
imagesc(radiance_map_tent_nosmooth(:, :, 1))
axis image;
title('Radiance Map: Tent Weighting and No Smoothness');
saveas(gcf, 'Results/radiance_map_tent_nosmooth.png')
hdrwrite(exp(radiance_map_tent_nosmooth), 'Results/hdr_image_tent_nosmooth.hdr')

%  No Weighting: using constant weighting function and full lambda 
[g_red_nowt]   = gsolve(Z(:, :, 1), B, lambda_full, no_weights);
[g_green_nowt] = gsolve(Z(:, :, 2), B, lambda_full, no_weights);
[g_blue_nowt]  = gsolve(Z(:, :, 3), B, lambda_full, no_weights);
plot_g(g_red_nowt, g_green_nowt, g_blue_nowt, 'Results/g_nowt_smooth.png');
radiance_map_nowt_smooth = get_radiance_map(file_names, [g_red_nowt g_green_nowt g_blue_nowt], no_weights, exposures);
clf(figure(3))
figure(3)
imagesc(radiance_map_nowt_smooth(:, :, 1))
axis image;
title('Radiance Map: No Weighting and Full Lambda');
saveas(gcf, 'Results/radiance_map_nowt_smooth.png')
hdrwrite(exp(radiance_map_nowt_smooth), 'Results/hdr_image_nowt_smooth.hdr')

% No Smoothness and No Weighting: using very low lambda and constant weighting
[g_red_nosmooth_nowt]   = gsolve(Z(:, :, 1), B, lambda_no_smooth, no_weights);
[g_green_nosmooth_nowt] = gsolve(Z(:, :, 2), B, lambda_no_smooth, no_weights);
[g_blue_nosmooth_nowt]  = gsolve(Z(:, :, 3), B, lambda_no_smooth, no_weights);
plot_g(g_red_nosmooth_nowt, g_green_nosmooth_nowt, g_blue_nosmooth_nowt, 'Results/g_nowt_nosmooth.png');
radiance_map_nowt_nosmooth = get_radiance_map(file_names, [g_red_nosmooth_nowt g_green_nosmooth_nowt g_blue_nosmooth_nowt], no_weights, exposures);
clf(figure(4))
figure(4)
imagesc(radiance_map_nowt_nosmooth(:, :, 1))
axis image;
title('Radiance Map: No Weighting and No Smoothness');
saveas(gcf, 'Results/radiance_map_nowt_nosmooth.png')
hdrwrite(exp(radiance_map_nowt_nosmooth), 'Results/hdr_image_nowt_nosmooth.hdr')

clear directory t i;