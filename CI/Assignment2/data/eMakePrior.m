function prior = eMakePrior(height, width)
    % A simple code to compute averaged power spectrum
    % Input: the prior size, and a set of images
    % Output: the expected/averaged power spectrum
	F2 = 0;
	count = 0;
    
    for p = 1:7
        
        strImg = ['./priors/', num2str(p), '.jpg'];
		f0 = im2double(imread(strImg));
		[height_L, width_L] = size(f0);

        for ky = 1:round(height/2):(height_L-height)

            for kx = 1:round(width/2):(width_L-width)

                F0 = fft2(f0(ky:(ky+height-1), kx:(kx+width-1)));
                F2 = F2 + F0 .* conj(F0);
		        count = count+1;            

            end 

        end 

    end

	prior = F2 / count;

end
