function [Z,B] = prepare_matrices_gsolve( ...
    file_names, ...
    exposures, ...
    sample_rate ...
    )
    
    %build the matrics B and Z
    %images is a list of images
    %exposures is the list of exposure, in the same order as images
    
    n_exposures = size(exposures,1);
    % Gget the number of pixels to sample
    tmp = imread(file_names{1});
    
    n_pixels = (fix(size(tmp,1)/sample_rate)+1) * (fix(size(tmp,2)/sample_rate)+1);
    
    Z = zeros(n_pixels,n_exposures,3);
    
    for k=1:n_exposures
        
        image = imread(file_names{k});
        index=1;
    
        for i=1:sample_rate:size(image,1)
        
            for j=1:sample_rate:size(image,2)
                
                Z(index,k,1) = image(i,j,1);
                Z(index,k,2) = image(i,j,2);
                Z(index,k,3) = image(i,j,3);
                index=index+1;

            end
        
        end
    
    end
    
    % Create exposure matrix
    B = zeros(n_exposures,1);

    for i = 1:n_exposures

        B(i) = log(exposures(i));

    end
    
end
