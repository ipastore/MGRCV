close all;  % Close all figures
clear variables;  % Clear all variables
clc;  % Clear command window




% Configuration parameters
root_data_path = '/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CI/Assignment4/data/';
root_output_path = '/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CI/Assignment4/output/';
if ~exist(root_output_path, 'dir')
    mkdir(root_output_path);
end
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part 1 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% % Display script starting message
% disp('=== NLOS Reconstruction Started ===');
% % Specify the data paths for the datasets 
% dataset_name1 = 'Z_d=0.5_l=[1x1]_s=[256x256]';
% dataset_name2 = 'planes_d=0.5_l=[16x16]_s=[16x16]';
% dataset_name3 = 'bunnybox_d=0.5_l=[16x16]_s=[16x16]';
% dataset_name4 = 'bunny_d=0.5_l=[16x16]_s=[16x16]';
% dataset_name5 = 'bunny_d=0.5_l=[1x1]_s=[256x256]';

% data_set_list = {dataset_name1, dataset_name2, dataset_name3, dataset_name4, dataset_name5};
% % data_set_list = {dataset_name4};  % Select individually for taking screenshots of some volumes

% % Define resolutions to test
% voxel_resolutions = [8, 16];  % Different voxel grid resolutions
% spad_samplings = [2, 4, 8];       % Different SPAD sampling rates

% % voxel_resolutions = [16];   % Select individually for taking screenshots of some volumes
% % spad_samplings = [8];       % Select individually for taking screenshots of some volumes

% % Initialize results table for performance tracking
% results = table();
% result_idx = 1;

% % Loop through each dataset
% for dataset_idx = 1:length(data_set_list)
%     dataset_name = data_set_list{dataset_idx};
%     data_path = fullfile(root_data_path, [dataset_name, '.mat']);
    
%     % Skip if dataset file doesn't exist
%     if ~exist(data_path, 'file')
%         warning('Dataset file not found: %s', data_path);
%         continue;
%     end
    
%     % Load measurement data
%     disp('==================================================');
%     fprintf('Loading dataset: %s\n', dataset_name);
%     measurement_data = load(data_path);
    
%     % Display dataset information
%     fprintf('Volume position: [%.2f, %.2f, %.2f]\n', measurement_data.data.volumePosition(1), ...
%         measurement_data.data.volumePosition(2), measurement_data.data.volumePosition(3));
%     fprintf('Volume size: %.2f\n', measurement_data.data.volumeSize);
    
%     % Create dataset-specific output folder
%     dataset_output_path = fullfile(root_output_path, dataset_name);
%     if ~exist(dataset_output_path, 'dir')
%         mkdir(dataset_output_path);
%     end
    
%     % Loop through different resolutions and sampling rates
%     for voxel_idx = 1:length(voxel_resolutions)
%         voxel_resolution = voxel_resolutions(voxel_idx);
        
%         for spad_idx = 1:length(spad_samplings)
%             spad_sampling = spad_samplings(spad_idx);
            
%             % Display current configuration
%             disp('--------------------------------------------------');
%             fprintf('Processing with voxel resolution: %d, SPAD sampling: %d\n', voxel_resolution, spad_sampling);
            
%             % Reconstruction phase
%             reconstruction_timer = tic;
%             reconstructed_volume = backprojection_naive(measurement_data.data, voxel_resolution, spad_sampling);
%             reconstruction_time = toc(reconstruction_timer);
            
%             % Report reconstruction performance
%             fprintf('Total reconstruction time: %.3f seconds\n', reconstruction_time);
            
%             % Save performance results
%             results(result_idx, :) = {dataset_name, voxel_resolution, spad_sampling, reconstruction_time};
%             result_idx = result_idx + 1;
            
%             % Apply Laplacian filter
%             disp('Applying Laplacian filter...');
%             filter_timer = tic;
%             f_lap = fspecial3('lap');
%             filtered_volume = imfilter(reconstructed_volume, -f_lap, 'symmetric');
%             filter_time = toc(filter_timer);
%             fprintf('Filter applied in %.3f seconds\n', filter_time);
            
%             % Save reconstructed volume
%             volume_file = fullfile(dataset_output_path, sprintf('volume_voxel%d_spad%d.mat', voxel_resolution, spad_sampling));
%             save(volume_file, 'reconstructed_volume', 'filtered_volume', 'voxel_resolution', 'spad_sampling', 'reconstruction_time', 'filter_time');
%             fprintf('Saved volume data to: %s\n', volume_file);
            
%             % For screenshot of individual volumes
%             % % 3D volume visualization
%             % volumeViewer(reconstructed_volume);
%             % volumeViewer(filtered_volume);
       
%         end
%     end
% end

% % Save performance results to CSV
% results.Properties.VariableNames = {'Dataset', 'VoxelResolution', 'SPADSampling', 'ReconstructionTime'};
% writetable(results, fullfile(root_output_path, 'performance_results.csv'));

% % Generate performance visualization
% figure('Name', 'Performance Comparison');
% datasets = unique(results.Dataset);
% markers = {'o', 's', 'd', '^', 'v', '>'};
% colors = lines(length(voxel_resolutions));

% for i = 1:length(datasets)
%     dataset_results = results(strcmp(results.Dataset, datasets{i}), :);
%     subplot(2, 3, i);
%     hold on;
    
%     for j = 1:length(voxel_resolutions)
%         vr_results = dataset_results(dataset_results.VoxelResolution == voxel_resolutions(j), :);
%         if ~isempty(vr_results)
%             plot(vr_results.SPADSampling, vr_results.ReconstructionTime, '-', 'Marker', markers{j}, ...
%                 'LineWidth', 2, 'Color', colors(j,:), 'DisplayName', sprintf('Voxel Res: %d', voxel_resolutions(j)));
%         end
%     end
    
%     title(sprintf('Dataset: %s', datasets{i}));
%     xlabel('SPAD Sampling Rate');
%     ylabel('Reconstruction Time (s)');
%     grid on;
%     legend('Location', 'NorthWest');
%     hold off;
% end

% % Save performance chart
% perf_chart_file = fullfile(root_output_path, 'performance_chart.png');
% saveas(gcf, perf_chart_file);
% fprintf('Saved performance chart to: %s\n', perf_chart_file);

% disp('=== All Processing Complete ===');

% % %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part 2 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% disp('================================================');
% disp('=== NLOS Confocal Reconstruction Started ===');

% % Configuration parameters for confocal reconstruction
% confocal_dataset_name = 'bunny_d=0.5_c=[256x256]';
% confocal_data_path = fullfile(root_data_path, [confocal_dataset_name, '.mat']);
% confocal_output_path = fullfile(root_output_path, confocal_dataset_name);

% if ~exist(confocal_output_path, 'dir')
%     mkdir(confocal_output_path);
% end

% % Load confocal measurement data
% fprintf('Loading confocal dataset: %s\n', confocal_dataset_name);
% confocal_data = load(confocal_data_path);

% % Display dataset information
% fprintf('Volume position: [%.2f, %.2f, %.2f]\n', confocal_data.data.volumePosition(1), ...
%     confocal_data.data.volumePosition(2), confocal_data.data.volumePosition(3));
% fprintf('Volume size: %.2f\n', confocal_data.data.volumeSize);

% % Define resolution parameters for confocal reconstruction
% confocal_voxel_resolution = 16;
% confocal_spad_sampling = 8;

% disp('--------------------------------------------------');
% fprintf('Processing confocal data with voxel resolution: %d, SPAD sampling: %d\n', ...
%     confocal_voxel_resolution, confocal_spad_sampling);

% % Reconstruction phase using confocal backprojection
% reconstruction_timer = tic;
% confocal_volume = backprojection_confocal(confocal_data.data, confocal_voxel_resolution, confocal_spad_sampling);
% confocal_reconstruction_time = toc(reconstruction_timer);

% % Report reconstruction performance
% fprintf('Confocal reconstruction time: %.3f seconds\n', confocal_reconstruction_time);

% % Apply Laplacian filter
% disp('Applying Laplacian filter to confocal reconstruction...');
% filter_timer = tic;
% f_lap = fspecial3('lap');
% confocal_filtered_volume = imfilter(confocal_volume, -f_lap, 'symmetric');
% confocal_filter_time = toc(filter_timer);
% fprintf('Filter applied in %.3f seconds\n', confocal_filter_time);

% % Save reconstructed volume
% confocal_volume_file = fullfile(confocal_output_path, ...
%     sprintf('confocal_volume_voxel%d_spad%d.mat', confocal_voxel_resolution, confocal_spad_sampling));
% save(confocal_volume_file, 'confocal_volume', 'confocal_filtered_volume', ...
%     'confocal_voxel_resolution', 'confocal_spad_sampling', 'confocal_reconstruction_time', 'confocal_filter_time');
% fprintf('Saved confocal volume data to: %s\n', confocal_volume_file);

% % % 3D volume visualization
% volumeViewer(confocal_volume);
% volumeViewer(confocal_filtered_volume);


% disp('=== All Processing Complete ===');

% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% Part 3 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
disp('================================================');
disp('=== NLOS Attenuation-Compensated Reconstruction Started ===');

% Configuration parameters for attenuation reconstruction
dataset_name1 = 'Z_d=0.5_l=[1x1]_s=[256x256]';
dataset_name2 = 'planes_d=0.5_l=[16x16]_s=[16x16]';

% Define resolution parameters
voxel_resolution = 32;
spad_sampling = 32;

% Process both datasets
for dataset_idx = 1:2
    % Get current dataset name
    if dataset_idx == 1
        dataset_name = dataset_name1;
    else
        dataset_name = dataset_name2;
    end
    
    % Set file paths
    data_path = fullfile(root_data_path, [dataset_name, '.mat']);
    dataset_output_path = fullfile(root_output_path, dataset_name);
    
    % Create output directory if it doesn't exist
    if ~exist(dataset_output_path, 'dir')
        mkdir(dataset_output_path);
    end
    
    % Load measurement data
    disp('--------------------------------------------------');
    fprintf('Loading dataset: %s\n', dataset_name);
    measurement_data = load(data_path);
    
    % Display dataset information
    fprintf('Volume position: [%.2f, %.2f, %.2f]\n', measurement_data.data.volumePosition(1), ...
        measurement_data.data.volumePosition(2), measurement_data.data.volumePosition(3));
    fprintf('Volume size: %.2f\n', measurement_data.data.volumeSize);
    
    % Reconstruction phase with attenuation compensation
    fprintf('Processing with voxel resolution: %d, SPAD sampling: %d\n', voxel_resolution, spad_sampling);
    
    % Apply attenuation-compensated backprojection
    reconstruction_timer = tic;
    reconstructed_volume = backprojection_attenuation(measurement_data.data, voxel_resolution, spad_sampling);
    reconstruction_time = toc(reconstruction_timer);
    
    % Report reconstruction performance
    fprintf('Attenuation-compensated reconstruction time: %.3f seconds\n', reconstruction_time);
    
    % Apply Laplacian filter
    disp('Applying Laplacian filter...');
    filter_timer = tic;
    f_lap = fspecial3('lap');
    filtered_volume = imfilter(reconstructed_volume, -f_lap, 'symmetric');
    filter_time = toc(filter_timer);
    fprintf('Filter applied in %.3f seconds\n', filter_time);
    
    % Save reconstructed volume
    volume_file = fullfile(dataset_output_path, sprintf('attenuation_volume_voxel%d_spad%d.mat', voxel_resolution, spad_sampling));
    save(volume_file, 'reconstructed_volume', 'filtered_volume', 'voxel_resolution', 'spad_sampling', 'reconstruction_time', 'filter_time');
    fprintf('Saved volume data to: %s\n', volume_file);
    
    % Display reconstructed volumes
    volumeViewer(reconstructed_volume);
    volumeViewer(filtered_volume);

end

disp('=== Attenuation-Compensated Reconstruction Complete ===');