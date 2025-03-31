function dataset = load_hdf5_dataset(fname)

    % read confocal flag from dataset file
    % (confocal data has c=[NxN] in the filename instead of l=[NxN]_s=[MxM]
    confocal = h5read(fname, '/isConfocal');
    
    dataset = struct;
   
    % spadOrigin: where the SPAD device is placed, faced towards the relay wall
    % spadPositions: SPAD measurement locations on the relay wall
    % spadNormals: normals of the relay wall at spadPositions
    
    dataset.spadPositions = h5read(fname, '/cameraGridPositions');
    dataset.spadNormals = h5read(fname, '/cameraGridNormals');
    dataset.spadOrigin = h5read(fname, '/cameraPosition');
    dataset.spadOrigin = dataset.spadOrigin(:);
     
    % laserOrigin: where the laser device is placed, faced towards the relay wall
    % laserPositions: locations on the relay wall illuminated by the laser
    % laserNormals: normals of the relay wall at laserPositions
    
    dataset.laserPositions = h5read(fname, '/laserGridPositions');
    dataset.laserNormals = h5read(fname, '/laserGridNormals');
    dataset.laserOrigin = h5read(fname, '/laserPosition');
    dataset.laserOrigin = dataset.laserOrigin(:);
    
    % volumePosition: center of the region of interest in the hidden scene
    % volumeSize: size of the region of interest in the hidden scene
    
    dataset.volumePosition = h5read(fname, '/hiddenVolumePosition')';
    dataset.volumePosition = dataset.volumePosition(:);
    dataset.volumeSize = h5read(fname, '/hiddenVolumeSize')';
    dataset.volumeSize = dataset.volumeSize(:);
    
    % data: impulse function H with five dimensions H(lu, lv, su, sv, t) where
    %       (lu, lv, su, sv) indicate horizontal (u) and vertical (v) indexing of
    %       the lasers and spads, corresponding to the indexing in the 
    %       spadPositions and laserPositions matrices. For confocal data, H
    %       only has 3 dimensions H(cu,cv,t), cu and cv correspond to
    %       co-located spad and laser positions (i.e. spadPositons, laserPositions)
    %       t indexes a measurement with time-of-flight (t-t0)*deltaT
    % t0: time shift (usually negative or zero) applied to all measurements 
    %     of the impulse function H. You need to subtract it from path lengths when
    %     calculating the indexing in the temporal dimension.
    %
    % deltaT: temporal resolution per pixel in the t dimension of H
    
    dataset.t0 = single(h5read(fname, '/t0'));
    dataset.deltaT = single(h5read(fname, '/deltaT'));
    
    data = h5read(fname, '/data');
    
    if confocal

        data = sum(data, 3);
        data = sum(data, 5);
        data = permute(data, [1 2 4 3 5]);
        data = data(:, :, :, 1, 1); 
    
    else

        data = sum(data, 5);
        data = sum(data, 7);
        data = permute(data, [1 2 3 4 6 5 7]);
        data = data(:, :, :, :, :, 1, 1); 
    
    end
    
    dataset.data = data;

end
