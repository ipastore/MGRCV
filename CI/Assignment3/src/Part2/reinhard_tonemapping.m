function [image] = reinhard_tonemapping( ...
    hdr, ...
    key, ...
    burn, ...
    fignum ...
    )

    eps=0.000001;
    log_average=exp(mean(log(hdr+eps),'all'));

    %scale everything so that mean = key
    image = key/log_average * hdr;
    
    I_white = burn*max(max(max(hdr)));
    burn_factor = 1+(image ./ (I_white^2));

    image = (image .* burn_factor) ./ (1+image);
    image = scale_and_gamma(image);

    clf(figure(fignum))
    figure(fignum)

    imagesc(image)
    axis image;

end
