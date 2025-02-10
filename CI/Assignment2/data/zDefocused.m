function f = zDefocused(f0, k, sigma, flag)

	[height, width] = size(f0);

    if flag == 1
        % Assume the defocus is circular convolution
		k = zPSFPad(k, height, width);
		k = fft2(fftshift(k));
		f = abs(ifft2(fft2(f0) .* k)) + randn(height, width) * sigma;

    else
        % Not make that assumption. But in this case, because of the 
        % boundary effects, the "effective" noise level is very high.
		k = rot90(k, 2);
		f = imfilter(f0, k) + randn(height, width) * sigma;
        
    end

end
