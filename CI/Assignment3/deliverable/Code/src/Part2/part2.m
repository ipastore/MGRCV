clear all;
clc;
close all;

fprintf('Starting HDR tone-mapping process...\n');

% Add necessary directories to MATLAB path
fprintf('Adding required directories to path...\n');
addpath('../Part1'); % For parse_files, prepare_matrices_gsolve, gsolve, etc.

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

% tone-map
keys = [0.09, 0.18, 0.36, 0.72];
burns = [0.1, 0.5, 1];

fprintf('Processing individual tone-mapped images...\n');

% First process and save individual images as before
for i = 1:length(hdr_images)
    hdr = hdr_images{i};
    img_name = image_names{i};
    fprintf('  Processing %s\n', img_name);
    
    % Process each combination of key and burn values
    for key_idx = 1:length(keys)
        key = keys(key_idx);
        for burn_idx = 1:length(burns)
            burn = burns(burn_idx);
            fprintf('    Key: %g, Burn: %g\n', key, burn);
            
            tone_mapped = reinhard_tonemapping(hdr, key, burn, 1);
            imwrite(tone_mapped, sprintf("Results/tonemapped_reinhard_key_%g_burn_%g_%s.png", key, burn, img_name));
        end
    end
    
    % Also create the simple HDR for each image
    fprintf('    Creating simple HDR...\n');
    simple_hdr = scale_and_gamma(hdr ./ (1 + hdr));
    imwrite(simple_hdr, sprintf("Results/simple_hdr_%s.png", img_name));
end

% Now create comparison figures for each key-burn pair
fprintf('\nCreating comparison figures for each key-burn combination...\n');
for key_idx = 1:length(keys)
    key = keys(key_idx);  % Fixed: using parentheses instead of brackets
    for burn_idx = 1:length(burns)
        burn = burns(burn_idx);  % Fixed: using parentheses instead of brackets
        fprintf('  Creating figure for Key: %g, Burn: %g\n', key, burn);
        
        % Create a 2x2 figure
        fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off');
        
        % Process each HDR image and add it to the figure
        for i = 1:length(hdr_images)
            fprintf('    Adding %s to figure\n', image_names{i});
            hdr = hdr_images{i};
            
            % Create subplot
            subplot(2, 2, i);
            
            % Tone map and display
            tone_mapped = reinhard_tonemapping(hdr, key, burn, 1);
            imshow(tone_mapped);
            
            % Add title
            title(titles{i}, 'FontSize', 12);
        end
        
        % Add overall title to the figure
        sgtitle(sprintf('Tone Mapping Comparison (Key: %g, Burn: %g)', key, burn), 'FontSize', 14);
        
        % Save the figure
        saveas(fig, sprintf("Results/comparison_key_%g_burn_%g.png", key, burn));
        close(fig);
    end
end

% Create a comparison figure for all simple HDR methods
fprintf('Creating comparison figure for simple HDR methods...\n');
fig = figure('Position', [100, 100, 1000, 800], 'Visible', 'off');
for i = 1:length(hdr_images)
    hdr = hdr_images{i};
    
    % Create subplot
    subplot(2, 2, i);
    
    % Create simple HDR and display
    simple_hdr = scale_and_gamma(hdr ./ (1 + hdr));
    imshow(simple_hdr);
    
    % Add title
    title(titles{i}, 'FontSize', 12);
end

% Add overall title to the figure
sgtitle('Simple HDR Comparison', 'FontSize', 14);

% Save the figure
saveas(fig, "Results/comparison_simple_hdr.png");

% Create a comprehensive figure for each method with a 4x3 grid for all key-burn combinations
fprintf('\nCreating comprehensive grid figures for each method with all key-burn combinations...\n');

for i = 1:length(hdr_images)
    hdr = hdr_images{i};
    img_name = image_names{i};
    method_title = titles{i};
    
    fprintf('  Creating comprehensive grid figure for %s...\n', img_name);
    
    % Create a large figure with 4 rows (keys) and 3 columns (burns)
    fig = figure('Position', [100, 100, 1200, 1000], 'Visible', 'off');
    
    % Counter for subplot positioning
    plot_idx = 1;
    
    % For each key-burn combination
    for key_idx = 1:length(keys)
        key = keys(key_idx);
        for burn_idx = 1:length(burns)
            burn = burns(burn_idx);
            
            fprintf('    Processing Key: %g, Burn: %g\n', key, burn);
            
            % Create subplot in grid
            subplot(length(keys), length(burns), plot_idx);
            
            % Tone map and display
            tone_mapped = reinhard_tonemapping(hdr, key, burn, 1);
            imshow(tone_mapped);
            
            % Add title with key and burn values
            title(sprintf('K:%.3g B:%.1f', key, burn), 'FontSize', 10);
            
            % Increment subplot counter
            plot_idx = plot_idx + 1;
        end
    end
    
    % Add overall title to the figure
    sgtitle(sprintf('All Key-Burn Combinations for %s', method_title), 'FontSize', 14);
    
    % Save the figure
    saveas(fig, sprintf("Results/all_combinations_%s.png", img_name));
    close(fig);
end

fprintf('HDR tone-mapping process completed successfully!\n');