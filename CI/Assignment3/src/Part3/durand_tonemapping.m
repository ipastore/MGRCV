function [final_image] = durand_tonemapping( ...
    hdr, ...
    dR, ...
    fignum ...
    )
    
    % Compute the intensity I by averaging the color channels.
    I = mean(hdr, 3);
    
    % Compute the log intensity: L=log2(I)
    L = log(I);
    
    % Filter that with a bilateral filter: B=bf(L)
    degreeOfSmoothing = 0.4;
    spatialSigma = 0.02 * size(hdr, 1);
    B = imbilatfilt(L, degreeOfSmoothing, spatialSigma);
    
    % Compute the detail layer: D=L−B
    D = L-B;
    
    clf(figure(fignum))
    figure(fignum)

    subplot(1, 3, 1)
    imagesc(L)
    axis image;
    
    subplot(1, 3, 2)
    imagesc(B)
    axis image;
    
    subplot(1, 3, 3)
    imagesc(D)
    axis image;
    
    saveas(gcf,"Results/base_decomposition.png")
    
    % Apply an offset and a scale to the base: B′=(B−o)∗s
    o = max(max(B));
    s = dR/(max(max(B)) - min(min(B)));
    B_prime = (B-o)*s;
    
    %Reconstruct the log intensity: O=2(B′+D)
    O = exp(B_prime+D);
    
    % Put back the colors: R′,G′,B′=O∗(R/I,G/I,B/I)
    final_image = zeros(size(hdr));
    
    for c=1:3

        final_image(:, :, c) = O .* (hdr(:, :, c) ./ I);
    
    end
    
    % Apply gamma compression
    final_image=scale_and_gamma(final_image);

end
