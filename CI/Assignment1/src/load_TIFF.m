%{
Autores: David Padilla Orenga, Ignacio Pastore Benaim
Asignatura: Computational Imaging
%}

clear
clc
close all

global plot_all;
plot_all = 0;
global deploy_all;
deploy_all = 0;

if deploy_all

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD TIFF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Load the image
    folderPath = '../data/images_tiff';
    % filename = 'bottles.tiff';
    filename = 'IMG_0596.tiff';
    % % filename = 'IMG_1026.tiff';
    % % filename = 'colors_noise.tiff';
    % filename = 'colors.tiff';
    fullFilePath = fullfile(folderPath, filename);
    img = imread(fullFilePath);

    % Retrieve image information
    info = imfinfo(fullFilePath);
    bitsPerPixel = info.BitDepth;
    width = info.Width;
    height = info.Height;

    % Report the image details
    fprintf('Bits per pixel: %d\n', bitsPerPixel);
    fprintf('Image width: %d pixels\n', width);
    fprintf('Image height: %d pixels\n', height);

    % Also we could retrieve the width and height and data type as:
    arrayWidth = size(img,2);
    arrayHeight = size(img,1);
    typeOfArray = class(img);


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LINEARIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Convert array of img to double
    imgDoubleArray = double(img);
    blackLevel = 1023; 
    saturationLevel = 15600; 

    % Linearize the image
    imgShifted = imgDoubleArray - blackLevel;                       % Shift the image so that blackLevel becomes 0
    imgLinearToClip = imgShifted / (saturationLevel - blackLevel);  % Scale the image so that saturationLevel becomes 1
    imgLinear = max(min(imgLinearToClip, 1), 0);                    % Clip values outside the range [0, 1]

    if plot_all
        % Display the linearized image
        figure;imshow(img);title('Linearized Image');uiwait;
    end


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BAYERN DEMOSAIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Choose pattern
    patterns = {'rggb', 'gbrg', 'grbg', 'bggr'};
    patternIndex = 1; %
    pattern = patterns{patternIndex};

    % % Apply Bayern demosaic with bilinear interpolation
    RGB_image = demosaic_bilinear(imgLinear, pattern);

    if plot_all
        % Display the demosaiced image with bilinear
        figure;imshow(RGB_image);title('Demosaiced Image with bilinear interpolation');uiwait;
    end

    % Apply Bayern demosaic with nearest neighbor and circshift
    RGB_image_nn_circshift = demosaic_nearest_neighbor(imgLinear, pattern);

    if plot_all
        % Display the demosaiced image with nearest neighbor
        figure;imshow(RGB_image_nn_circshift);title('Demosaiced Image with nearest neighbor');uiwait;
    end

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% WHITE BALANCING %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Apply white world white balancing 
    balancedImage_WW = whiteWorldWB(RGB_image);
    % Apply gray world white balancing
    balancedImage_GW = grayWorldWB(RGB_image);
    % Apply manual white balancing
    figure, imshow(RGB_image); title('Choose a pixel')
    refPoint = round(ginput(1));
    uiwait;

    balancedImage_manual = manualWhiteBalance(RGB_image, refPoint);

    if plot_all
        % Display the results
        figure; imshow(balancedImage_WW); title('White World Assumption');
        figure; imshow(balancedImage_GW); title('Gray World Assumption');
        figure; imshow(balancedImage_manual); title('Manual White Balance');
        uiwait;
    end

    % Select the final balanced image
    final_balancedImage = balancedImage_manual;
    % final_balancedImage = balancedImage_GW;

    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DENOISER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    kernel_size = 3;
    denoisedImage_mean = denoiser_mean(final_balancedImage, kernel_size);
    denoisedImage_median = denoiser_median(final_balancedImage, kernel_size);
    denoisedImage_gaussian = denoiser_gaussian(final_balancedImage, kernel_size, 1);



    if plot_all
        figure; imshow(final_balancedImage); title('Without denoising');
        figure; imshow(denoisedImage_mean); title('Denoised Image with mean filter');
        figure; imshow(denoisedImage_median); title('Denoised Image with median filter');
        figure; imshow(denoisedImage_gaussian); title('Denoised Image with gaussian filter');
        uiwait;
    end

    save('denoisedImage_gaussian.mat', 'denoisedImage_gaussian');


    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COLOR BALANCE %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % load('./denoisedImage_gaussian.mat', 'denoisedImage_gaussian');

    % Boost saturation with different levels
    if plot_all
        saturation_factors = [1, 1.25, 1.5, 1.75, 2];
        for i = 1:length(saturation_factors)
            img_color_balanced = color_balance(denoisedImage_gaussian, saturation_factors(i));
            figure;
            imshow(img_color_balanced);
            title(sprintf('Saturation factor: %.2f', saturation_factors(i)));
        end
        uiwait;
    end


    % Select desired saturation
    selected_saturation_factor = 1.50;

    img_color_balanced = color_balance(denoisedImage_gaussian, selected_saturation_factor);
    save('img_color_balanced.mat', 'img_color_balanced');

    if plot_all
        figure;imshow(img_color_balanced);title(sprintf('Color balanced image with saturation factor %.2f', selected_saturation_factor));uiwait;
    end



    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% TONE REPRODUCTION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

    % Load the image
    % load('./img_color_balanced.mat', 'img_color_balanced');
    
    if plot_all 
        % Apply alpha correction for different values and display the figures
        porcentage_brighten = [0.25, 0.50, 0.75];
        for i = 1:length(porcentage_brighten)
            img_alpha = alpha_correction(img_color_balanced, porcentage_brighten(i));
            gamma_values = [1.7, 1.8, 1.9, 2, 2.2, 2.4];
            for j = 1:length(gamma_values)
                img_gamma = tone_reproduction(img_alpha, gamma_values(j));
                figure;
                imshow(img_gamma);
                title(sprintf('Gamma corrected image with gamma %.2f and porcentage: %.2f', gamma_values(j), porcentage_brighten(i)));
            end
        end
    
        for i = length(porcentage_brighten)
            img_gamma = tone_reproduction(img_color_balanced, porcentage_brighten(i));
            figure;
            imshow(img_gamma);
            title(sprintf('Gamma corrected image with gamma %.2f', gamma_values(j)));
        end
    
    end
    
    selected_porcentage_brighten = 0.75;
    img_alpha = alpha_correction(img_color_balanced, selected_porcentage_brighten);
    
    selected_gamma = 1.8; 
    img_gamma = tone_reproduction(img_color_balanced, selected_gamma);
    figure;imshow(img_gamma);title(sprintf('Gamma corrected with %.2f and brighten porcentage %.2f', selected_gamma, selected_porcentage_brighten));uiwait;
save('img_gamma.mat', 'img_gamma');
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% COMPRESSION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

load('./img_gamma.mat', 'img_gamma');

% Save as PNG (lossless)
imwrite(img_gamma, 'output_image.png', 'PNG');

% Get file information for the PNG file
png_info = dir('output_image.png');

qualities = [95, 90, 85, 80, 75, 70, 65, 60, 55, 50, 45, 40, 35, 30, 25, 20, 15, 10, 5];
compression_ratios = zeros(size(qualities));

for i = 1:length(qualities)
    quality = qualities(i);
    output_filename = sprintf('output_image_%d.jpg', quality);
    
    % Save as JPEG with the current quality
    imwrite(img_gamma, output_filename, 'Quality', quality);
    
    % Compute compression ratios
    jpeg_info = dir(output_filename);
    compression_ratios(i) = png_info.bytes / jpeg_info.bytes;
    
    fprintf('Compression Ratio (PNG to JPEG, Quality %d): %.2f\n', quality, compression_ratios(i));
end


% Compression Ratio (PNG to JPEG, Quality 95): 5.66
% Compression Ratio (PNG to JPEG, Quality 90): 8.99
% Compression Ratio (PNG to JPEG, Quality 85): 12.08
% Compression Ratio (PNG to JPEG, Quality 80): 14.76
% Compression Ratio (PNG to JPEG, Quality 75): 17.28
% Compression Ratio (PNG to JPEG, Quality 70): 19.14
% Compression Ratio (PNG to JPEG, Quality 65): 20.91
% Compression Ratio (PNG to JPEG, Quality 60): 22.65
% Compression Ratio (PNG to JPEG, Quality 55): 24.03
% Compression Ratio (PNG to JPEG, Quality 50): 25.26
% Compression Ratio (PNG to JPEG, Quality 45): 26.48
% Compression Ratio (PNG to JPEG, Quality 40): 28.08
% Compression Ratio (PNG to JPEG, Quality 35): 29.59
% Compression Ratio (PNG to JPEG, Quality 30): 31.54
% Compression Ratio (PNG to JPEG, Quality 25): 33.87
% Compression Ratio (PNG to JPEG, Quality 20): 36.77
% Compression Ratio (PNG to JPEG, Quality 15): 40.16
% Compression Ratio (PNG to JPEG, Quality 10): 44.75
% Compression Ratio (PNG to JPEG, Quality 5): 50.56
% Hasta 75 no se ve perdida

%% ---------------------------- FUNCTIONS --------------------------------%
function img_alpha = alpha_correction(img_color_balanced, porcentage_brighten)

    % Brighten the image 
    max_gray = max(max(rgb2gray(img_color_balanced))); % Get maximum grayscale value
    exposure_alpha = log2(porcentage_brighten / max_gray); % Scale to % of max brightness

    % Apply alpha correction
    img_alpha = img_color_balanced .* 2^exposure_alpha;;
    
    % Clip values to [0, 1]
    img_alpha = min(max(img_alpha, 0), 1);
end

function img_gamma = tone_reproduction(img_linear, gamma_value)
    
    % Apply sRGB gamma correction
    img_gamma = zeros(size(img_linear));
    threshold = 0.0031308;
    
    % Piecewise function for each channel
    for c = 1:3 % R, G, B channels
        linear = img_linear(:,:,c);
        nonlinear = zeros(size(linear));
        
        % Case 1: linear <= threshold
        mask_low = linear <= threshold;
        nonlinear(mask_low) = 12.92 .* linear(mask_low);
        
        % Case 2: linear > threshold
        mask_high = ~mask_low;
        nonlinear(mask_high) = (1.055) .* (linear(mask_high).^(1/gamma_value)) - 0.055;
        
        img_gamma(:,:,c) = nonlinear;
    end
    
    % Clip final values
    img_gamma = min(max(img_gamma, 0), 1);
end

function img_out = color_balance(img_rgb, saturation_factor)
    % Convert RGB to HSV
    img_hsv = rgb2hsv(img_rgb);
    
    % Boost saturation channel (S is the 2nd channel)
    img_hsv(:,:,2) = img_hsv(:,:,2) .* saturation_factor;
    
    % Clip values to [0, 1]
    img_hsv(:,:,2) = min(max(img_hsv(:,:,2), 0), 1);
    
    % Convert back to RGB
    img_out = hsv2rgb(img_hsv);
end


function [R, G, B] = demosaic_bayer(imgLinear, pattern)

    % Initialize the color channels
    [height, width] = size(imgLinear);
    R = zeros(height, width);
    G = zeros(height, width);
    B = zeros(height, width);

    % Apply the Bayer demosaic depending on the pattern
    switch lower(pattern)
        case 'grbg'
            G(1:2:end, 1:2:end) = imgLinear(1:2:end, 1:2:end); % Green
            R(1:2:end, 2:2:end) = imgLinear(1:2:end, 2:2:end); % Red
            B(2:2:end, 1:2:end) = imgLinear(2:2:end, 1:2:end); % Blue
            G(2:2:end, 2:2:end) = imgLinear(2:2:end, 2:2:end); % Green

        case 'rggb'
            R(1:2:end, 1:2:end) = imgLinear(1:2:end, 1:2:end); % Red
            G(1:2:end, 2:2:end) = imgLinear(1:2:end, 2:2:end); % Green
            G(2:2:end, 1:2:end) = imgLinear(2:2:end, 1:2:end); % Green
            B(2:2:end, 2:2:end) = imgLinear(2:2:end, 2:2:end); % Blue

        case 'bggr'
            B(1:2:end, 1:2:end) = imgLinear(1:2:end, 1:2:end); % Blue
            G(1:2:end, 2:2:end) = imgLinear(1:2:end, 2:2:end); % Green
            G(2:2:end, 1:2:end) = imgLinear(2:2:end, 1:2:end); % Green
            R(2:2:end, 2:2:end) = imgLinear(2:2:end, 2:2:end); % Red

        case 'gbrg'
            G(1:2:end, 1:2:end) = imgLinear(1:2:end, 1:2:end); % Green
            B(1:2:end, 2:2:end) = imgLinear(1:2:end, 2:2:end); % Blue
            R(2:2:end, 1:2:end) = imgLinear(2:2:end, 1:2:end); % Red
            G(2:2:end, 2:2:end) = imgLinear(2:2:end, 2:2:end); % Green

        otherwise
            error('Unknown Bayer pattern: %s', pattern);
    end
end

function RGB_image = demosaic_bilinear(imgLinear, pattern)
    
    %Kernel
    kernel = ones(3,3);

    % Apply the Bayer pattern
    [R, G, B] = demosaic_bayer(imgLinear, pattern);

    % Make the channel masks
    [height, width] = size(imgLinear);
    Phantom_ones = ones(height, width);
    [R_mask, G_mask, B_mask] = demosaic_bayer(Phantom_ones, pattern);

    % Get denomiators
    R_denominator = conv2(R_mask, kernel, 'same');
    G_denominator = conv2(G_mask, kernel, 'same');
    B_denominator = conv2(B_mask, kernel, 'same');

    % Get numeratos
    R_numerator = conv2(R, kernel, 'same');
    G_numerator = conv2(G, kernel, 'same');
    B_numerator = conv2(B, kernel, 'same');

    % Divide and get the RGB image
    R = R_numerator ./ R_denominator;
    G = G_numerator ./ G_denominator;
    B = B_numerator ./ B_denominator;
    RGB_image = cat(3, R, G, B);

end

function RGB_image = demosaic_nearest_neighbor(imgLinear, pattern)
    % Separate the RAW image into R, G, B channels based on Bayer pattern
    [R, G, B] = demosaic_bayer(imgLinear, pattern);
    
    % Interpolate each channel
    R_interp = nearest_neighbor_interpolation(R);
    G_interp = nearest_neighbor_interpolation(G);
    B_interp = nearest_neighbor_interpolation(B);
    
    RGB_image = cat(3, R_interp, G_interp, B_interp);
end

function channel_interp = nearest_neighbor_interpolation(channel)
    [height, width] = size(channel);
    known_mask = (channel ~= 0);        % Identify known pixels
    channel_interp = channel;           % Initialize output

    % Direction shifts: Right, Left, Down, Up
    shifts = [0, 1; 0, -1; 1, 0; -1, 0];

    % Loop until all missing pixels are filled
    while any(~known_mask(:))
        prev_mask = known_mask;

        for k = 1:size(shifts, 1)
            dx = shifts(k, 1);
            dy = shifts(k, 2);

            % Shift known mask and the channel
            shifted_mask = circshift(prev_mask, [dx, dy]);
            shifted_channel = circshift(channel_interp, [dx, dy]);

            % Find missing pixels adjacent to known pixels
            fill_mask = shifted_mask & ~known_mask;

            % Fill missing pixels
            channel_interp(fill_mask) = shifted_channel(fill_mask);

            % Update the known mask
            known_mask = known_mask | fill_mask;
        end
    end
end

function balancedImage = whiteWorldWB(inputImage)
    
    % Find the maximum value in each channel (Two max needed: one for row and another for column)
    maxR = max(max(inputImage(:,:,1)));
    maxG = max(max(inputImage(:,:,2)));
    maxB = max(max(inputImage(:,:,3)));
    
    % Compute scaling factors based on the highest intensity
    scaleR = maxG / maxR;
    scaleB = maxG / maxB;
    % No need to scale the green channel since it is the reference
        
    % Application of the scale factor for each of the channels
    balancedImage(:,:,1) = inputImage(:,:,1) * scaleR;
    balancedImage(:,:,2) = inputImage(:,:,2);
    balancedImage(:,:,3) = inputImage(:,:,3) * scaleB;

end

function balancedImage = grayWorldWB(inputImage)
    
    % Compute the mean intensity of each channel (Two max needed: one for row and another for column)
    meanR = mean(mean(inputImage(:,:,1)));
    meanG = mean(mean(inputImage(:,:,2)));
    meanB = mean(mean(inputImage(:,:,3)));
    
    % Compute scaling factors based on the green channel (as reference)
    scaleR = meanG / meanR;
    scaleG = 1;  % Green is used as reference
    scaleB = meanG / meanB;
    
    % Application of the scale factor for each of the channels
    balancedImage(:,:,1) = inputImage(:,:,1) * scaleR;
    balancedImage(:,:,2) = inputImage(:,:,2) * scaleG;
    balancedImage(:,:,3) = inputImage(:,:,3) * scaleB;
    
end

function balancedImage = manualWhiteBalance(inputImage, refPoint)
    
    % Coordinates of the reference point
    row = refPoint(1);
    col = refPoint(2);
    
    % We get the RGB values of the point
    R_ref = inputImage(row, col, 1);
    G_ref = inputImage(row, col, 2);
    B_ref = inputImage(row, col, 3);
    
    % Computation of scale factors as the pdf says
    scaleR = (R_ref + G_ref + B_ref) / (3 * R_ref);
    scaleG = (R_ref + G_ref + B_ref) / (3 * G_ref);
    scaleB = (R_ref + G_ref + B_ref) / (3 * B_ref);
    
    % Application of the scale factor for each of the channels
    balancedImage(:,:,1) = inputImage(:,:,1) * scaleR;
    balancedImage(:,:,2) = inputImage(:,:,2) * scaleG;
    balancedImage(:,:,3) = inputImage(:,:,3) * scaleB;
    
end

function denoised_image = denoiser_mean(inputImage, kernel_size)
    
    %Kernel
    kernel = ones(kernel_size,kernel_size);

    % Get the RGB values. Each channel will have different noise levels
    R = inputImage(:,:,1);
    G = inputImage(:,:,2);
    B = inputImage(:,:,3);

    % Make the channel masks
    [height, width] = size(R);
    Phantom_ones = ones(height, width);

    % Get numeratos
    R_numerator = conv2(R, kernel, 'same');
    G_numerator = conv2(G, kernel, 'same');
    B_numerator = conv2(B, kernel, 'same');
    % Get denomiator
    Denominator = conv2(Phantom_ones, kernel, 'same');

    % Divide and get the RGB image
    R = R_numerator ./ Denominator;
    G = G_numerator ./ Denominator;
    B = B_numerator ./ Denominator;
    denoised_image = cat(3, R, G, B);

end

function denoised_image = denoiser_median(inputImage, kernel_size)

    % Obtener dimensiones
    [h, w, c] = size(inputImage);
    pad_size = floor(kernel_size / 2);

    % Inicializar imagen de salida
    denoised_image = zeros(h, w, c);

    % Aplicar filtro de mediana en cada canal sin `padarray`
    for ch = 1:c
        % Extraer canal actual
        channel = inputImage(:,:,ch);

        % Crear una imagen ampliada con padding (sin `padarray`)
        padded_channel = zeros(h + 2*pad_size, w + 2*pad_size);

        % Copiar la imagen en el centro de la matriz ampliada
        padded_channel(pad_size+1:end-pad_size, pad_size+1:end-pad_size) = channel;

        % Replicar los bordes (Simulando 'replicate' de `padarray`)
        % Bordes superior e inferior
        padded_channel(1:pad_size, :) = padded_channel(pad_size+1:2*pad_size, :);
        padded_channel(end-pad_size+1:end, :) = padded_channel(end-2*pad_size+1:end-pad_size, :);

        % Bordes izquierdo y derecho
        padded_channel(:, 1:pad_size) = padded_channel(:, pad_size+1:2*pad_size);
        padded_channel(:, end-pad_size+1:end) = padded_channel(:, end-2*pad_size+1:end-pad_size);

        % Crear una matriz de ventanas concatenadas
        windows = cat(3, ...
            padded_channel(1:end-2, 1:end-2), padded_channel(1:end-2, 2:end-1), padded_channel(1:end-2, 3:end), ...
            padded_channel(2:end-1, 1:end-2), padded_channel(2:end-1, 2:end-1), padded_channel(2:end-1, 3:end), ...
            padded_channel(3:end, 1:end-2), padded_channel(3:end, 2:end-1), padded_channel(3:end, 3:end));

        % Calcular la mediana en la tercera dimensión (los 9 valores por píxel)
        denoised_image(:,:,ch) = median(windows, 3);
    end

end

function kernel = gaussian_kernel(kernel_size, sigma)
    % Crear una malla de coordenadas centrada en (0,0)
    [x, y] = meshgrid(-floor(kernel_size/2):floor(kernel_size/2), -floor(kernel_size/2):floor(kernel_size/2));

    % Aplicar la ecuación de la función Gaussiana
    kernel = exp(-(x.^2 + y.^2) / (2 * sigma^2)) / (2 * pi * sigma^2);

    % Normalizar para que la suma de los valores sea 1
    kernel = kernel / sum(kernel(:));
end

function denoised_image = denoiser_gaussian(inputImage, kernel_size, sigma)
    
    %Kernel
    kernel = gaussian_kernel(kernel_size, sigma);

    % Get the RGB values. Each channel will have different noise levels
    R = inputImage(:,:,1);
    G = inputImage(:,:,2);
    B = inputImage(:,:,3);

    % Get numeratos
    R = conv2(R, kernel, 'same');
    G = conv2(G, kernel, 'same');
    B = conv2(B, kernel, 'same');
    denoised_image = cat(3, R, G, B);

end
