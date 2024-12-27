import os
import numpy as np

seq_name = "Seq_035"
type = "flexible_exhaustive"
inputfile = "f_0082858_f_0082968_matches.npz"
outputfile = inputfile
npz_dir = f"./data/npz_outputs/"

npz_path = os.path.join(npz_dir, seq_name, type, inputfile)

# parts = inputfile.split("_")
# image1 = parts[:2]
# image2 = parts[2:4]

# image1 = "_".join(image1)
# image2 = "_".join(image2)

output_path = os.path.join(npz_dir, seq_name, type, outputfile)

data = np.load(npz_path)

kpts0 = data["keypoints0"]
kpts1 = data["keypoints1"]
mkpts0_inlier_idx = np.array([])
mkpts1_inlier_idx = np.array([])


# Dictionary to hold the keypoints, matches, match confidence, and descriptors
out_matches = {
    'keypoints0': kpts0,  
    'keypoints1': kpts1,
    'inliers0_idx': mkpts0_inlier_idx,
    'inliers1_idx': mkpts1_inlier_idx,
}

# Close the file
data.close()

# Save the updated dictionary with inliers
np.savez(output_path, **out_matches)
print(f"Matches saved to {output_path}")


