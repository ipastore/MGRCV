function G = backprojection_naive(data, resolution_voxel, resolution_capture)
    % Extract scene parameters
    laserOrigin = data.laserOrigin;
    spadOrigin = data.spadOrigin;
    laserPositions = data.laserPositions;
    spadPositions = data.spadPositions;
    volumePosition = data.volumePosition;
    volumeSize = data.volumeSize;
    
    % Create a 3D voxel grid
    [i, j, k] = ndgrid(1:resolution_voxel, 1:resolution_voxel, 1:resolution_voxel);
    
    % Calculate the center coordinates for each voxel
    voxel_size = volumeSize / resolution_voxel;
    x = volumePosition(1) - volumeSize/2 + (i - 0.5) * voxel_size;
    y = volumePosition(2) - volumeSize/2 + (j - 0.5) * voxel_size;
    z = volumePosition(3) - volumeSize/2 + (k - 0.5) * voxel_size;
    
    % Initialize the reconstruction volume
    G = zeros(resolution_voxel, resolution_voxel, resolution_voxel);
    
    % Loop over all laser positions
    for l_i = 1:size(laserPositions, 1)
        for l_j = 1:size(laserPositions, 2)
            % Get current laser position
            laser_pos = squeeze(laserPositions(l_i, l_j, :));
            
            % Compute distance from laser origin to laser position (d1)
            d1 = norm(laser_pos - laserOrigin);
            
            % Loop over all SPAD positions (possibly subsampled)
            for s_i = 1:resolution_capture:size(spadPositions, 1)
                for s_j = 1:resolution_capture:size(spadPositions, 2)
                    % Get current SPAD position
                    spad_pos = squeeze(spadPositions(s_i, s_j, :));
                    
                    % Compute distance from SPAD origin to SPAD position (d4)
                    d4 = norm(spad_pos - spadOrigin);
                    
                    % For all voxels:
                    for a = 1:resolution_voxel
                        for b = 1:resolution_voxel
                            for c = 1:resolution_voxel
                                % Get current voxel position
                                voxel_pos = [x(a,b,c), y(a,b,c), z(a,b,c)];
                                
                                % Compute distance from voxel to laser position (d2)
                                d2 = norm(voxel_pos - laser_pos');
                                
                                % Compute distance from voxel to SPAD position (d3)
                                d3 = norm(voxel_pos - spad_pos');
                                
                                % Compute total time-of-flight
                                t = d1 + d2 + d3 + d4;
                                
                                % Convert to discrete time index
                                idx = round(t / data.deltaT) + data.t0;
                                
                                % Bounds checking for index
                                if idx >= 1 && idx <= size(data.data, 5)
                                    % Accumulate intensity value
                                    G(a,b,c) = G(a,b,c) + data.data(l_i, l_j, s_i, s_j, idx);
                                end
                            end
                        end
                    end
                end
            end
            
            % Print progress
            fprintf('Processed laser position (%d,%d)\n', l_i, l_j);
        end
    end
end