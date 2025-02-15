clear all;
clc;
close all;

% Choose the output folder
output_folder = '../output';

% Define sigma values and blur sizes
sigma_values = [0.001, 0.005, 0.01, 0.02];  % Gaussian noise levels
blur_sizes = [3, 7, 15];  % Different blur sizes
% Define apertures
apertures = {'zhou', 'raskar', 'Levin', 'circular'};

% Read and preprocess image
image = imread('images/penguins.jpg');
image = image(:, :, 1);
f0 = im2double(image);
[height, width, ~] = size(f0);

% Allow cancellation: use Ctrl+C to interrupt the script
try
    for aperture_name = apertures
        aperture = imread(['apertures/', aperture_name{1}, '.bmp']);
        for sigma = sigma_values
            for blurSize = blur_sizes
                fprintf('Testing with aperture = %s, sigma = %.4f, blurSize = %d\n', aperture_name{1}, sigma, blurSize);

                % Generate prior matrix
                A_star = eMakePrior(height, width) + 1e-8;
                C = sigma.^2 * height * width ./ A_star;

                % Create blur kernel
                temp = fspecial('disk', blurSize);
                flow = max(temp(:));
                k1 = im2double(imresize(aperture, [2*blurSize + 1, 2*blurSize + 1], 'nearest'));
                k1 = k1 * (flow / max(k1(:)));

                % Apply blur
                f1 = zDefocused(f0, k1, sigma, 0);

                % Recover using Wiener deconvolution without prior
                f0_hat_wnr_wout_prior = deconvwnr(f1, k1, sigma^2);
                convolutionType = 'Wiener_wout_prior';
                save_results(f0, f1, f0_hat_wnr_wout_prior, aperture_name{1}, sigma, blurSize, convolutionType, output_folder);
                analyze_power_spectrum(k1, f0, f1, aperture_name{1}, sigma, blurSize, convolutionType, output_folder);


                % Recover using Wiener deconvolution with prior
                f0_hat_wnr = zDeconvWNR(f1, k1, C);
                convolutionType = 'Wiener';
                save_results(f0, f1, f0_hat_wnr, aperture_name{1}, sigma, blurSize, convolutionType, output_folder);
                analyze_power_spectrum(k1, f0, f1, aperture_name{1}, sigma, blurSize, convolutionType, output_folder);


                % Recover using Lucy-Richardson Deconvolution
                for n_it = [5, 10, 20]
                    f0_hat_lucy = deconvlucy(f1, k1, n_it);
                    convolutionType = sprintf('Lucy_%d', n_it);
                    save_results(f0, f1, f0_hat_lucy, aperture_name{1}, sigma, blurSize, convolutionType, output_folder);
                    analyze_power_spectrum(k1, f0, f1, aperture_name{1}, sigma, blurSize, convolutionType, output_folder);

                end

            end
        end
    end
catch ME
    if strcmp(ME.identifier, 'MATLAB:ExecutionInterrupted')
        disp('Execution cancelled by user.');
        return;
    else
        rethrow(ME);
    end
end

function save_results(f0, f1, f0_hat, aperture_name, sigma, blurSize, convolutionType, output_folder)
    % Create invisible figure 
    fig = figure('Visible','off');
    subplot(1, 3, 1), imshow(f0), title('Focused');
    subplot(1, 3, 2), imshow(f1), title(sprintf('Defocused (sigma=%.3f, blurSize=%d)', sigma, blurSize));
    subplot(1, 3, 3), imshow(f0_hat), title(sprintf('Recovered (%s)', convolutionType));
    output_filename = sprintf('sigma%.3f_blurSize%d.png', sigma, blurSize);
    fullPath = fullfile(output_folder, aperture_name, convolutionType, sprintf('sigma%.3f_blurSize%d', sigma, blurSize));
    if ~exist(fullPath, 'dir')
        mkdir(fullPath);
    end
    saveas(fig, fullfile(fullPath, output_filename));
    close(fig);
end

function analyze_power_spectrum(k1, f0, f1, aperture_name, sigma, blurSize, convolutionType, output_folder)
    % Padding aperture
    [height, width] = size(f0);
    k1P = zPSFPad(k1, max(height, width), max(height, width));

    % Aperture power spectra
    F = fft2(k1P);
    F = fftshift(F .* conj(F));
    S = log(F);
    S_X = S(:, round(length(S)/2) + 1);
    S_Y = S(round(length(S)/2) + 1, :);

    % Display results
    fig1 = figure('Visible','off');
    subplot_tight(2, 2, 1, 0.05, false)
    imagesc(k1P);
    axis('image')
    axis('off')
    title('Aperture');
    subplot_tight(2, 2, 2, 0.05, false)
    imagesc(S);
    axis('image')
    axis('off')
    title('Aperture Frequency');
    subplot_tight(2, 2, 3, 0.05, false)
    plot(linspace(-1, 1, length(S_X)), S_X)
    grid('on')
    title('Normalized frequency X');
    subplot_tight(2, 2, 4, 0.05, false)
    plot(linspace(-1, 1, length(S_Y)), S_Y)
    grid('on')
    title('Normalized frequency Y');
    output_filename = sprintf('spectrum_aperture_sigma%.3f_blurSize%d.png', sigma, blurSize);
    fullPath = fullfile(output_folder, aperture_name, convolutionType, sprintf('sigma%.3f_blurSize%d', sigma, blurSize));
    if ~exist(fullPath, 'dir')
        mkdir(fullPath);
    end
    saveas(fig1, fullfile(fullPath, output_filename));
    close(fig1);

    % Image power spectra
    F0 = fft2(f0);
    F0 = fftshift(F0 .* conj(F0));
    F1 = fft2(f1);
    F1 = fftshift(F1 .* conj(F1));
    S0 = log(F0);
    S0_X = S0(:, round(length(S0)/2) + 1);
    S0_Y = S0(round(length(S0)/2) + 1, :);
    S1 = log(F1);
    S1_X = S1(:, round(length(S1)/2) + 1);
    S1_Y = S1(round(length(S1)/2) + 1, :);

    % Display results
    fig2 = figure('Visible','off');
    subplot_tight(2, 2, 1, [0.1 0.05], false)
    plot(linspace(-1, 1, length(S0_X)), S0_X)
    grid('on')
    title('Original X');
    subplot_tight(2, 2, 2, [0.1 0.05], false)
    plot(linspace(-1, 1, length(S0_Y)), S0_Y)
    grid('on')
    title('Original Y');
    subplot_tight(2, 2, 3, [0.1 0.05], false)
    plot(linspace(-1, 1, length(S1_X)), S1_X)
    grid('on')
    title('Defocused X');
    subplot_tight(2, 2, 4, [0.1 0.05], false)
    plot(linspace(-1, 1, length(S1_Y)), S1_Y)
    grid('on')
    title('Defocused Y');
    output_filename = sprintf('spectrum_image_sigma%.3f_blurSize%d.png', sigma, blurSize);
    saveas(fig2, fullfile(fullPath, output_filename));
    close(fig2);
end

function outK = zPSFPad(inK, height, width)
    % This function is to zeropadding the psf
    [sheight, swidth] = size(inK);
    outK = zeros(height, width);
    outK(floor(end/2-sheight/2) + 1:floor(end/2-sheight/2) + sheight, ...
         floor(end/2-swidth/2) + 1:floor(end/2-swidth/2) + swidth) = inK;
end