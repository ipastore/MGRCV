function [scaled] = scale_and_gamma(image)
    
    min_value = min(min(min(image)));
    max_value = max(max(max(image)));
    scaled = (image - min_value) / (max_value - min_value);
    
    % min_value = min(min(image));
    % max_value = max(max(image));
    % scaled = zeros(size(image));
    % scaled(:,:,1) = (image(:,:,1) - min_value(1))/(max_value(1) - min_value(1));
    % scaled(:,:,2) = (image(:,:,2) - min_value(2))/(max_value(2) - min_value(2));
    % scaled(:,:,3) = (image(:,:,3) - min_value(3))/(max_value(3) - min_value(3));

    scaled = scaled .^ (1/2.2);

end
