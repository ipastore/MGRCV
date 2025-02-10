function f0 = zDeconvWNR(f, k, C)
    % This is the Weiner deconvolution algorithm using 1/f law
    % f: defocused image
    % k: defocus kernel
    % C: sigma^2/A

	[height, width] = size(f);

	k = zPSFPad(k, height, width);
	k = fft2(fftshift(k));

	f0 = abs(ifft2((fft2(f) .* conj(k)) ./ (k .* conj(k) + C)));

end

function outK = zPSFPad(inK, height, width)
    % This function is to zeropadding the psf
    
	[sheight, swidth] = size(inK);

	outK = zeros(height, width);
	
    outK(floor(end/2-sheight/2) + 1:floor(end/2-sheight/2) + sheight, ...
         floor(end/2-swidth/2) + 1:floor(end/2-swidth/2) + swidth) = inK;

end
