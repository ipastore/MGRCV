%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% LOAD TIFF %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Define the folder path and filename
folderPath = '../data/images_tiff';
% filename = 'bottles.tiff';
filename = 'IMG_0596.tiff';

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
figure;
imshow(imgLinear);
title('Linearized Image');


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% DEMOSAIC %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%It could be done with demosaic()



