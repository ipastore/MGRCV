% Read stack of images with their exposure
directory = ('../Data/'); % change dir to run a different stack

[file_names, exposures] = parse_files(directory);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve for camera response function
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% tent weight function
t = 1:128; 

% weights 
tent_weights = cat(2, t, t(end:-1:1)); 

%no weight function
no_weights = ones(256);

% lamda smoothing factor (default=50)
lambda = 50; 

% Load and sample the images
[Z, B] = prepare_matrices_gsolve(file_names, exposures, 20);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Solve for camera response function and plot
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
weights = tent_weights;
%weights = no_weights;

% Solve for g in each color channel
[g_red] = gsolve(Z(:, :, 1), B, lambda, weights);

[g_green] = gsolve(Z(:, :, 2), B, lambda, weights);

[g_blue] = gsolve(Z(:, :, 3), B, lambda, weights);

% Compute the hdr radiance map (Section 2.2 in Debevec)

% Plot recovered g
plot_g(g_red, g_green, g_blue, "Results/g_tent_smooth.png");

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Compute and plot radiance map
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
radiance_map = get_radiance_map(file_names, [g_red g_green g_blue], weights, exposures);

clf(figure(2))
figure(2)
imagesc(radiance_map(:, :, 1))
axis image;

saveas(gcf, "Results/radiance_map.png")

hdrwrite(exp(radiance_map), 'Results/hdr_image.hdr')

clear directory t i;