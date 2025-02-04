%{
Autores: David Padilla Orenga, Ignacio Pastore Benaim
Asignatura: Computational Imaging
%}

clear
clc
close all

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD TIFF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the image
folderPath = '../data/images_tiff';
% filename = 'bottles.tiff';
filename = 'IMG_0596.tiff';
% filename = 'IMG_1026.tiff';
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

% % Display the linearized image
figure;
imshow(imgLinear);
title('Linearized Image');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% BAYERN DEMOSAIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Choose pattern
patterns = {'rggb', 'gbrg', 'grbg', 'bggr'};
patternIndex = 1; %
pattern = patterns{patternIndex};

% % Apply Bayern demosaic with bilinear interpolation
RGB_image = demosaic_bilinear(imgLinear, pattern);

% Display the demosaiced image with bilinear
figure;
imshow(RGB_image);
title('Demosaiced Image with bilinear interpolation');


% Apply Bayern demosaic with nearest neighbor and circshift
RGB_image_nn_circshift = demosaic_nearest_neighbor(imgLinear, pattern);
figure;
imshow(RGB_image_nn_circshift);
title('Demosaiced Image with NN')


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
balancedImage_manual = manualWhiteBalance(RGB_image, refPoint);

% Display the results
figure; imshow(balancedImage_WW); title('White World Assumption');
figure; imshow(balancedImage_GW); title('Gray World Assumption');
figure; imshow(balancedImage_manual); title('Manual White Balance');




%% ---------------------------- FUNCTIONS --------------------------------%

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


