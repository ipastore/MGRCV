# Computer Vision - Final Assignment
# Ignacio Pastore Benaim: 920576

This repository contains the deliverables for the final assignment of the Computer Vision course. Below is a description of the organization of the repository and a brief guide on how to use the provided code.

## Glossary
- "more_flexible": indicating matches with RANSAC reprojection error of 4.
- "exhaustive": indicating exhaustive pairing of all images.
- "undistorted": indicating preprocessing of image.
- "toy": indicating a toy (short) version for the project. Specifically for Seq_035

### Note
Some absolute paths should be modified in the utils files in order to work. Haven´t also tested other paths.

## Folders

- `data/`: Excluded to reduce size file. (Sending if requested)
    - `npz_outputs/`: Outputs of selected npz files from Superglue outputs.
    - `Seq_027/`: All images used in the project: raw, resized and undistorted.
    - `Seq_035/`: All imags used in the project.
    - `calib.yaml/`: Calibration file for the camera, slightly modified.
- `colmap_projects/`: 3 Colmap projects.
- `own_projects/`: 2 fully implemented projects.
- `Superglue_outputs/`: data folder containing all images of keypoints, matches and inliers. Also npz files that were copied to the data/ folder. Excluded to reduce size file. (Sending if requested)
- `utils/`: 
    - `SuperGluePretrainedNetwork/`: Excluded to reduce size file. (Sending if requested)
        - `match_pairs_RANSAC_openCV/`: Modified match_pairs.py file with RANSAC using open CV library. Moved to utils upfolder to deliver.
        - `match_pairs_RANSAC_own/`:  Modified match_pairs.py file with in-house implemented functions. Moved to utils upfolder to deliver.
    - `colmap_preprocess.py`: Script for processing the output of Superglue to a colmap project.
    - `comparison_pnp_approaches.py`: Script for compairing own implementation of PnP vs open CV.
    - `comparison_undistort_fish_eye.py`: Script for compairing own implementation of undistort fish eye vs open CV.
    - `cv_plot_bundle_sqlite_helpers_functions.py`: Helper functions for my_own_COLMAP.py.
    - `image_preprocess.py`:Script for resizing and undistort images from a fish eye lense.
    - `own_colmap_preprocess.py`: Preprocessing of npz outputs to own colmap implementation.
    - `own_database_preprocess.py`: Script for building the database from prepared npz outputs to an own colmap project.




## Usage Guide

1. **Preprocess images**:
With image_preprocess.py

2. **Find match pairs with match_pairs_RANSAC_own**:
You can modify the RANSAC reprojection error inside the file. (It´s commented)

3. **Process npz output to own project**:
With own_colmap_preprocess

4. **Prepare database**:
With own_database_preprocess.py

5. **Run my_own_COLMAP.py**:
Inside, modify relative paths and imaages order.
 

