# HDR Image Processing

This project implements HDR image reconstruction and tone mapping techniques using MATLAB.

## Project Structure

- `src/Part1/`: HDR reconstruction with various weighting functions. (Uses images from the `Data/` folder.)
- `src/Part2/`: Reinhard tone mapping (global operator).
- `src/Part3/`: Local tone mapping methods (Durand and naive approaches).
- `src/Part4/`: Custom HDR pipeline for a different image source. (Uses images from the `Data_1/` folder; images must be named with their exposure times.)
- `Data/`: Contains the Memorial Church HDR dataset for Part1.
- `Data_1/`: Contains your custom HDR dataset for Part4.

## Running the Code

Each part can be run independently:

```matlab
% From within each Part directory:
part1  % Reads images from Data/, applies HDR reconstruction (using step=20 for sampling)
part2  % Executes Reinhard tone mapping on HDR images
part3  % Compares local tone mapping approaches (Durand and naive methods)
part4  % Processes images from Data_1/ (ensure images are named with exposure times)
```

Results are saved in each part's `Results/` subdirectory.

## Key Design Choices

### HDR Reconstruction (Part1)
- Implements downsampling with a user-defined step (step=20 is used) to control memory usage.
- Uses different weighting functions, with "tent" weighting yielding the best compromise.
- Employs lambda (λ=50) for smoothing during camera response estimation.

### Tone Mapping (Part2-3)
- **Reinhard Operator:** A global operator that adjusts mid-tones (key) and highlights (burn) for realistic images.
- **Local Operators:** Durand uses bilateral filtering to separate the base and detail layers; the naive method applies direct contrast reduction.

### Custom Processing (Part4)
- Designed for a different dataset (Data_1/) where images must be pre-named with their exposure times.
- Automatically adds required directories to the MATLAB path.

## Parameters

- **HDR Reconstruction (Part1):** Uses tent weights, smoothing factor λ=50, and sampling step=20.
- **Reinhard Tone Mapping (Part2):** Typical parameters are key=0.18 (mid-gray) and burn=0.5 (for highlights).
- **Durand (Part3):** Uses a dynamic range compression factor dR=4 and adapts spatial filtering based on image size.
