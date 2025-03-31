function G = backProjection_attenuation_clean(data, resolution_voxel, spad_step)

    % Extract scene information
    laser_origin = data.laserOrigin;
    spad_origin = data.spadOrigin;
    laser_positions = data.laserPositions;
    spad_positions = data.spadPositions;
    volume_center = data.volumePosition;
    volume_size = data.volumeSize;

    % Voxel grid setup
    [i, j, k] = ndgrid(1:resolution_voxel, 1:resolution_voxel, 1:resolution_voxel);
    step = volume_size / resolution_voxel;
    xv = volume_center(1) - volume_size/2 + (i - 0.5) * step;
    yv = volume_center(2) - volume_size/2 + (j - 0.5) * step;
    zv = volume_center(3) - volume_size/2 + (k - 0.5) * step;

    % Flatten voxel coordinates
    voxel_coords = [xv(:)'; yv(:)'; zv(:)'];
    G = zeros(size(xv));

    for l_i = 1:size(laser_positions,1)
        for l_j = 1:size(laser_positions,2)
            l = squeeze(laser_positions(l_i, l_j, :));
            d1 = norm(l - laser_origin(:));
            n_l = squeeze(data.laserNormals(l_i, l_j, :));

            for s_i = 1:size(spad_positions,1)
                for s_j = 1:size(spad_positions,2)
                    s = squeeze(spad_positions(s_i, s_j, :));
                    d4 = norm(s - spad_origin(:));
                    n_s = squeeze(data.spadNormals(s_i, s_j, :));

                    % Voxel distances
                    vec_l = bsxfun(@minus, voxel_coords, l);
                    vec_s = bsxfun(@minus, voxel_coords, s);
                    d2 = sqrt(sum(vec_l.^2, 1));
                    d3 = sqrt(sum(vec_s.^2, 1));
                    tof = d1 + d2 + d3 + d4;

                    % Cosine terms
                    cos_l = dot(vec_l, repmat(n_l,1,size(vec_l,2))) ./ (d2 * norm(n_l));
                    cos_s = dot(vec_s, repmat(n_s,1,size(vec_s,2))) ./ (d3 * norm(n_s));

                    % Total attenuation correction
                    attenuation = (cos_l .* cos_s) ./ (d2 .* d3 + eps);

                    % Time index
                    idx = round(tof / data.deltaT) + data.t0;
                    valid = idx >= 1 & idx <= size(data.data,5);

                    if any(valid)
                        H = zeros(1, numel(idx));
                        H(valid) = data.data(l_i, l_j, s_i, s_j, idx(valid));
                        G = G + reshape(H .* attenuation, size(G));
                    end
                end
            end
        end
    end
end