function [radiance_map] = get_radiance_map( ...
    file_names, ...
    g_funcs, ...
    weights, ...
    exposures ...
    )

    n_exposures = size(exposures,1);
    tmp = imread(file_names{1});

    radiance_map = zeros(size(tmp));
    weights_sum = zeros(size(tmp));

    for k=1:n_exposures

        image = imread(file_names{k});
        exp=log(exposures(k));

        for i=1:size(image, 1)

            for j=1:size(image, 2)

                for c=1:3

                    Z = image(i, j, c) + 1;

                    w = weights(Z);

                    radiance_map(i, j, c) = radiance_map(i, j, c) + w * (g_funcs(Z, c) - exp);

                    weights_sum(i, j, c) = weights_sum(i, j, c) + w;

                end

            end

        end

    end

    radiance_map = radiance_map ./ weights_sum;

end
