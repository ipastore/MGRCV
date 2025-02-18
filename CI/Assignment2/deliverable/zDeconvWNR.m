% function f0 = zDeconvWNR(f, k, C)
%     % This is the Weiner deconvolution algorithm using 1/f law
%     % f: defocused image
%     % k: defocus kernel
%     % C: sigma^2/A

% 	[height, width] = size(f);

% 	k = zPSFPad(k, height, width);
% 	k = fft2(fftshift(k));

% 	f0 = abs(ifft2((fft2(f) .* conj(k)) ./ (k .* conj(k) + C)));

% end

function f0 = zDeconvWNR(f, k, C)

    [height, width, channels] = size(f);

    kPadded = zPSFPad(k, height, width);
    kFFT = fft2(fftshift(kPadded));

    % Replicate the kernel to match the 3rd dimension.
    kRep = repmat(kFFT, 1, 1, channels);
    denom = (kRep .* conj(kRep) + C);

    fFFT = fft2(f);  % 2D FFT per channel
    numerator = fFFT .* conj(kRep);

    f0 = abs(ifft2(numerator ./ denom));
end

function outK = zPSFPad(inK, height, width)
    % This function is to zeropadding the psf
    
	[sheight, swidth] = size(inK);

	outK = zeros(height, width);
	
    outK(floor(end/2-sheight/2) + 1:floor(end/2-sheight/2) + sheight, ...
         floor(end/2-swidth/2) + 1:floor(end/2-swidth/2) + swidth) = inK;

end
