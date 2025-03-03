function [final_image] = naive_contrast_reduction(hdr, dR, fignum, img_name, img_title)
    
    % Default image name if not provided
    if nargin < 4
        img_name = '';
    end
    
    % Default image title if not provided
    if nargin < 5
        img_title = img_name;
    end
    
    method_name = 'naive';
    
    % Compute the intensity I by averaging the color channels.
    I = mean(hdr, 3);
    
    % Compute the log intensity: L=log(I)
    L = log(I);
    
    % In naive approach, we don't separate base and detail layers
    % Instead apply contrast reduction directly to the log image
    
    % Apply an offset and a scale directly to log image
    o = max(max(L));
    s = dR/(max(max(L)) - min(min(L)));
    L_prime = (L-o)*s;
    
    % Reconstruct the intensity
    O = exp(L_prime);
    
    % Put back the colors: R′,G′,B′=O∗(R/I,G/I,B/I)
    final_image = zeros(size(hdr));
    
    for c=1:3
        final_image(:, :, c) = O .* (hdr(:, :, c) ./ I);
    end
    
    % Apply gamma compression
    final_image = scale_and_gamma(final_image);
    
    % Optional visualization
    if nargin > 2
        % Create figure without displaying it
        fig = figure('visible', 'off');
        figure(fig);
        
        subplot(1, 2, 1)
        imagesc(L)
        axis image;
        title('Log Intensity');
        
        subplot(1, 2, 2)
        imagesc(L_prime)
        axis image;
        title('Contrast Reduced');
        
        % Add overall title with method name and image title
        sgtitle(sprintf('Naive Contrast Reduction - %s (dR = %d)', img_title, dR), 'FontSize', 14);
        
        % Save with unique filename including the method, image name and dynamic range
        save_filename = sprintf("Results/decomposition_%s_dR_%d_%s.png", method_name, dR, img_name);
        saveas(gcf, save_filename);
    end
end
