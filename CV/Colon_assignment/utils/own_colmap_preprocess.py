import os
import numpy as np

def prepare_colmap_files_with_inliers(npz_files, output_dir):
    """
    Process .npz files to create COLMAP-compatible .txt files for keypoints and inliers.

    Args:
        npz_files (list): List of paths to .npz files.
        output_dir (str): Directory where output .txt files will be saved.

    Returns:
        None
    """
    # Create output directories if they don't exist
    keypoints_dir = os.path.join(output_dir, "keypoints")
    matches_dir = os.path.join(output_dir, "matches")
    os.makedirs(keypoints_dir, exist_ok=True)
    os.makedirs(matches_dir, exist_ok=True)



    for npz_file in npz_files:
        # Load the .npz file
        data = np.load(npz_file)

        # Extract file names from the npz filename
        base_name = os.path.basename(npz_file).replace(".npz", "")
        parts = base_name.split("_")
        image1 = "_".join(parts[:-3])
        image2 = "_".join(parts[2:4])

        print(f"Processing {image1}_{image2}")

        # Process keypoints for image1
        keypoints0 = data["keypoints0"]  # Adjust origin
        keypoints0_npz = os.path.join(keypoints_dir, f"{image1}")
        save_keypoints(keypoints0, keypoints0_npz)

        # Process keypoints for image2
        keypoints1 = data["keypoints1"]   # Adjust origin
        keypoints1_npz = os.path.join(keypoints_dir, f"{image2}")
        save_keypoints(keypoints1, keypoints1_npz)

      # Extract inliers indeices from npz file: extract two nd arrays with the title mkpts0_inliers_idx and mkpts1_inliers_idx and then hstack
        mkpts0_inliers_idx = data["inliers0_idx"]
        mkpts1_inliers_idx = data["inliers1_idx"]
        inlier_indices = np.column_stack((mkpts0_inliers_idx, mkpts1_inliers_idx))
        matches_npz = os.path.join(matches_dir, f"{image1}_{image2}_inliers")
        F = data["F"]
        save_matches(inlier_indices, image1, image2, F, matches_npz)

    print(f"Inlier-based files prepared and saved in {output_dir}")


def save_keypoints(keypoints, filepath):
    """
    Save keypoints in .npz format.

    Args:
        keypoints (np.ndarray): Array of keypoints (N x 2).
        filepath (str): Path to the output .npz file.

    Returns:
        None
    """
    np.savez(filepath, keypoints=keypoints)


def save_matches(matches, image1, image2, F,filepath):
    """
    Save matches in .npz format.

    Args:
        matches (np.ndarray): Array of matches (N x 2).
        filepath (str): Path to the output .npz file.

    Returns:
        None
    """
    np.savez(filepath, matches=matches, image1=image1, image2=image2, F=F)


if __name__ == "__main__":

    # Name of Seq
    seq_name = "Seq_035"
    type = "toy_own"

    # Directory containing .npz files
    npz_dir = f"../data/npz_outputs/{seq_name}/{type}"
    output_dir = f"../colmap_prepared/{seq_name}/{type}"

    # List all .npz files in the directory
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]

    # Prepare COLMAP files
    prepare_colmap_files_with_inliers(npz_files, output_dir)