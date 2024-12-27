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
    all_inliers_path = os.path.join(matches_dir, "all_inliers.txt")



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
        keypoints0 = data["keypoints0"] + 0.5  # Adjust origin
        keypoints0_txt = os.path.join(keypoints_dir, f"{image1}.png.txt")
        save_keypoints(keypoints0, keypoints0_txt)

        # Process keypoints for image2
        keypoints1 = data["keypoints1"] + 0.5  # Adjust origin
        keypoints1_txt = os.path.join(keypoints_dir, f"{image2}.png.txt")
        save_keypoints(keypoints1, keypoints1_txt)

      # Extract inliers indeices from npz file: extract two nd arrays with the title mkpts0_inliers_idx and mkpts1_inliers_idx and then hstack
        mkpts0_inliers_idx = data["inliers0_idx"]
        mkpts1_inliers_idx = data["inliers1_idx"]
        inlier_indices = np.column_stack((mkpts0_inliers_idx, mkpts1_inliers_idx))
        matches_txt = os.path.join(matches_dir, f"{image1}_{image2}_inliers.txt")
        save_matches(inlier_indices, image1, image2, matches_txt)

        # Save all inliers in a single file
        with open(all_inliers_path, "a") as f:
            f.write(f"{image1}.png {image2}.png\n")
            for match in inlier_indices:
                f.write(f"{int(match[0])} {int(match[1])}\n")
            f.write("\n")  # Add an empty line to separate different image pairs

    print(f"Inlier-based files prepared and saved in {output_dir}")


def save_keypoints(keypoints, filepath):
    """
    Save keypoints in COLMAP-compatible format.

    Args:
        keypoints (np.ndarray): Array of keypoints (N x 2).
        filepath (str): Path to the output .txt file.

    Returns:
        None
    """
    with open(filepath, "w") as f:
        f.write(f"{len(keypoints)} 128\n")  # Number of keypoints and descriptor size
        for kp in keypoints:
            f.write(f"{kp[0]:.6f} {kp[1]:.6f} 0.0 0.0 " + " ".join(["0"] * 128) + "\n")


def save_matches(matches, image1, image2, filepath):
    """
    Save matches in COLMAP-compatible format.

    Args:
        matches (np.ndarray): Array of matches (N x 2).
        filepath (str): Path to the output .txt file.

    Returns:
        None
    """
    with open(filepath, "w") as f:
        f.write(f"{image1}.png {image2}.png\n")  # Header with image filenames
        for match in matches:
            f.write(f"{int(match[0])} {int(match[1])}\n")


# Example usage
if __name__ == "__main__":

    # Name of Seq
    seq_name = "Seq_035"
    type = "flexible_sequential"

    # Directory containing .npz files
    npz_dir = f"./data/npz_outputs/{seq_name}/{type}"
    output_dir = f"./colmap_prepared/{seq_name}/{type}"

    # List all .npz files in the directory
    npz_files = [os.path.join(npz_dir, f) for f in os.listdir(npz_dir) if f.endswith(".npz")]

    # Prepare COLMAP files
    prepare_colmap_files_with_inliers(npz_files, output_dir)