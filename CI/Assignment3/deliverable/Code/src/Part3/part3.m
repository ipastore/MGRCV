clear all;
clc;
close all;

fprintf('Starting local tone mapping comparison process...\n');

% Add necessary directories to MATLAB path
fprintf('Adding required directories to path...\n');
addpath('../Part1'); % For parse_files, prepare_matrices_gsolve, gsolve, etc.
addpath('../Part2'); % For reinhard_tonemapping


% Set all figures to be invisible
set(0, 'DefaultFigureVisible', 'off');

% load HDR
fprintf('Loading HDR images...\n');
nowt_nosmooth = hdrread('../Part1/Results/hdr_image_nowt_nosmooth.hdr');
nowt_smooth = hdrread('../Part1/Results/hdr_image_nowt_smooth.hdr');
tent_nosmooth = hdrread('../Part1/Results/hdr_image_tent_nosmooth.hdr');
tent_smooth = hdrread('../Part1/Results/hdr_image_tent_smooth.hdr');

% Store all HDR images in a cell array with their names
hdr_images = {nowt_nosmooth, nowt_smooth, tent_nosmooth, tent_smooth};
image_names = {'nowt_nosmooth', 'nowt_smooth', 'tent_nosmooth', 'tent_smooth'};
titles = {'No Weight + No Smooth', 'No Weight + Smooth', 'Tent + No Smooth', 'Tent + Smooth'};

fignum = 6;

% Create Results directory if it doesn't exist
if ~exist('Results', 'dir')
    fprintf('Creating Results directory...\n');
    mkdir('Results');
end

% Process with both methods for different dynamic range values
fprintf('\nProcessing images with both tone mapping methods...\n');
dR_values = [2, 4, 6, 8];

for i = 1:length(hdr_images)
    hdr = hdr_images{i};
    img_name = image_names{i};
    img_title = titles{i};
    fprintf('\n  Processing %s (%s)\n', img_name, img_title);
    
    for dR_idx = 1:length(dR_values)
        dR = dR_values(dR_idx);
        fprintf('    Dynamic Range: %d\n', dR);
        
        % Apply Durand tonemapping
        fprintf('      Applying Durand tonemapping...\n');
        durand_img = durand_tonemapping(hdr, dR, fignum, img_name, img_title);
        imwrite(durand_img, sprintf("Results/tonemapped_durand_dR_%d_%s.png", dR, img_name));
        fignum = fignum + 1;
        
        % Apply naive contrast reduction
        fprintf('      Applying naive contrast reduction...\n');
        naive_img = naive_contrast_reduction(hdr, dR, fignum, img_name, img_title);
        imwrite(naive_img, sprintf("Results/naive_contrast_reduction_dR_%d_%s.png", dR, img_name));
        fignum = fignum + 1;
        
        % Create a side-by-side comparison
        fprintf('      Creating side-by-side comparison...\n');
        figure('visible', 'off');
        subplot(1, 2, 1);
        imshow(naive_img);
        title(sprintf('Naive (dR = %d)', dR));
        
        subplot(1, 2, 2);
        imshow(durand_img);
        title(sprintf('Durand (dR = %d)', dR));
        
        sgtitle(sprintf('Tone Mapping Comparison - %s', img_title), 'FontSize', 14);
        saveas(gcf, sprintf("Results/comparison_dR_%d_%s.png", dR, img_name));
    end
    
    % Create a comprehensive comparison figure for this HDR image with all dR values
    fprintf('    Creating comprehensive comparison for %s...\n', img_name);
    figure('visible', 'off', 'Position', [100, 100, 1200, 800]);
    
    for dR_idx = 1:length(dR_values)
        dR = dR_values(dR_idx);
        
        % Load naive image
        naive_img = imread(sprintf("Results/naive_contrast_reduction_dR_%d_%s.png", dR, img_name));
        subplot(2, 4, dR_idx);
        imshow(naive_img);
        title(sprintf('Naive (dR = %d)', dR));
        
        % Load durand image
        durand_img = imread(sprintf("Results/tonemapped_durand_dR_%d_%s.png", dR, img_name));
        subplot(2, 4, dR_idx + 4);
        imshow(durand_img);
        title(sprintf('Durand (dR = %d)', dR));
    end
    
    sgtitle(sprintf('Comparison of Tone Mapping Methods - %s', img_title), 'FontSize', 14);
    saveas(gcf, sprintf("Results/comprehensive_comparison_%s.png", img_name));
end

% Create a grid comparison for each dynamic range value across all HDR techniques
fprintf('\nCreating method comparison grids for each dynamic range...\n');
for dR_idx = 1:length(dR_values)
    dR = dR_values(dR_idx);
    fprintf('  Creating grid for dynamic range: %d\n', dR);
    
    % Create figure for Naive method
    figure('visible', 'off', 'Position', [100, 100, 1000, 800]);
    for i = 1:length(hdr_images)
        subplot(2, 2, i);
        img = imread(sprintf("Results/naive_contrast_reduction_dR_%d_%s.png", dR, image_names{i}));
        imshow(img);
        title(titles{i}, 'FontSize', 12);
    end
    sgtitle(sprintf('Naive Contrast Reduction (dR = %d)', dR), 'FontSize', 14);
    saveas(gcf, sprintf("Results/naive_comparison_dR_%d.png", dR));
    
    % Create figure for Durand method
    figure('visible', 'off', 'Position', [100, 100, 1000, 800]);
    for i = 1:length(hdr_images)
        subplot(2, 2, i);
        img = imread(sprintf("Results/tonemapped_durand_dR_%d_%s.png", dR, image_names{i}));
        imshow(img);
        title(titles{i}, 'FontSize', 12);
    end
    sgtitle(sprintf('Durand Tonemapping (dR = %d)', dR), 'FontSize', 14);
    saveas(gcf, sprintf("Results/durand_comparison_dR_%d.png", dR));
end

fprintf('\nLocal tone mapping comparison process completed successfully!\n');
