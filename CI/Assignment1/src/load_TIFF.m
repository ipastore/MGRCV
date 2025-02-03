%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD TIFF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the folder path and filename
folderPath = '../data/images_tiff';
% filename = 'bottles.tiff';
filename = 'IMG_0596.tiff';
% filename = 'IMG_1026.tiff';

% Construct the full file path
fullFilePath = fullfile(folderPath, filename);

% Load the image
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

% Display
% figure;
% imshow(img);
% title('Before Linearization');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LINEARIZATION %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Convert array of img to double
imgDoubleArray = double(img);

blackLevel = 1023; 
saturationLevel = 15600; 

% Shift the image so that blackLevel becomes 0
imgShifted = imgDoubleArray - blackLevel;

% Scale the image so that saturationLevel becomes 1
imgLinearToClip = imgShifted / (saturationLevel - blackLevel);

% Clip values outside the range [0, 1]
imgLinear = max(min(imgLinearToClip, 1), 0);


% Display the linearized image
% figure;
% imshow(imgLinear);
% title('Linearized Image');

% Apply Bayern demosaic
pattern = 'rggb';
% pattern = 'gbrg';
% pattern = 'grbg';
% pattern = 'bggr';

RGB_image = demosaic_bilinear(imgLinear, pattern);
figure;
imshow(RGB_image);
title('Demosaiced Image with bilinear interpolation');




%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEMOSAIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It could be done with demosaic()

function [R, G, B] = demosaic_bayer(imgLinear, pattern)
    % Ensure the input is double
    imgLinear = double(imgLinear);

    % Get the size of the image
    [height, width] = size(imgLinear);

    % Initialize the color channels
    R = zeros(height, width);
    G = zeros(height, width);
    B = zeros(height, width);

    % Apply the Bayer pattern
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

    % Ensure the input is double
    imgLinear = double(imgLinear);

    % Get the size of the image
    [height, width] = size(imgLinear);

    % Initialize the color channels
    Phantom_ones = ones(height, width);

    % Apply the Bayer pattern
    [R, G, B] = demosaic_bayer(imgLinear, pattern);

    % Make masks
    [R_mask, G_mask, B_mask] = demosaic_bayer(Phantom_ones, pattern);

    % Get denomiators
    R_denominator = conv2(R_mask, kernel, 'same');
    G_denominator = conv2(G_mask, kernel, 'same');
    B_denominator = conv2(B_mask, kernel, 'same');

    % Get numeratos
    R_numerator = conv2(R, kernel, 'same');
    G_numerator = conv2(G, kernel, 'same');
    B_numerator = conv2(B, kernel, 'same');

    % Divide
    R = R_numerator ./ R_denominator;
    G = G_numerator ./ G_denominator;
    B = B_numerator ./ B_denominator;

    RGB_image = cat(3, R, G, B);

end

    