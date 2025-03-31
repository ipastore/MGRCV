function G = backprojection_confocal(data, resolution_voxel, resolution_capture)
    % Extract scene parameters
    laserOrigin = data.laserOrigin;
    spadOrigin = data.spadOrigin;
    
    % laserPositions equal to Spad positions in confocal case
    positions = data.laserPositions;
    
    volumePosition = data.volumePosition;
    volumeSize = data.volumeSize;
    
    % Create a 3D grid of voxel indices
    [i, j, k] = ndgrid(1:resolution_voxel, 1:resolution_voxel, 1:resolution_voxel);
    
    % Calculate the center coordinates for each voxel
    voxel_centers = zeros(3, resolution_voxel^3);
    voxel_centers(1,:) = volumePosition(1) - volumeSize/2 + (i(:)' - 0.5) * (volumeSize / resolution_voxel);
    voxel_centers(2,:) = volumePosition(2) - volumeSize/2 + (j(:)' - 0.5) * (volumeSize / resolution_voxel);
    voxel_centers(3,:) = volumePosition(3) - volumeSize/2 + (k(:)' - 0.5) * (volumeSize / resolution_voxel);
    
    % Initialize the reconstruction volume
    G = zeros(resolution_voxel, resolution_voxel, resolution_voxel);
    
    % Process confocal measurements
    for l_i = 1:resolution_capture:size(positions, 1)
        for l_j = 1:resolution_capture:size(positions, 2)
            % Get current laser/SPAD position
            pos = squeeze(positions(l_i, l_j, :));
            
            % Distance from the laser to the relay wall
            d1 = norm(pos - laserOrigin);
            
            % Distance from the relay wall to the SPAD (same as d1 in confocal case)
            d4 = norm(spadOrigin - pos);
            
            % Vectorized distance calculation from each voxel to the laser/SPAD position
            diff_vecs = bsxfun(@minus, voxel_centers, pos);
            d2 = vecnorm(diff_vecs, 2, 1);
            d3 = d2;  % In confocal case, these distances are the same
            
            % Compute total time-of-flight
            t_total = d1 + d2 + d3 + d4;
            
            % Convert to discrete time indices
            indices = round(t_total / data.deltaT) + data.t0;
            
            % Bounds checking for indices
            valid_indices = indices >= 1 & indices <= size(data.data, 3);
            
            % Accumulate intensity values for valid indices
            for idx = 1:length(indices)
                if valid_indices(idx)
                    % Convert linear index back to 3D index for G
                    [a, b, c] = ind2sub([resolution_voxel, resolution_voxel, resolution_voxel], idx);
                    G(a, b, c) = G(a, b, c) + data.data(l_i, l_j, indices(idx));
                end
            end
        end
        
        % Print progress
        fprintf('Processed row %d/%d\n', l_i, size(positions, 1));
    end
    
    % Reshape to ensure proper 3D volume output
    G = reshape(G, [resolution_voxel, resolution_voxel, resolution_voxel]);
end