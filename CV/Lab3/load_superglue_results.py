import numpy as np


path = './SuperGlue_output/image1_image2_matches.npz'
npz = np.load(path) # Load dictionary with super point

# Create a boolean mask with True for keypoints with a good match, and False for the rest
mask = npz['matches'] > -1
# Using the boolean mask, select the indexes of matched keypoints from image 2
idxs = npz['matches'][mask]
# Using the boolean mask, select the keypoints from image 1 with a good match
x1_sp = npz['keypoints0'][mask]
# Using the indexes, select matched keypoints from image 2
x2_sp = npz['keypoints1'][idxs]

# Now, x1_sp and x2_sp contain the matched keypoints from image 1 and image 2, respectively
print("Matched keypoints from image 1:", x1_sp)
print("Matched keypoints from image 2:", x2_sp)
