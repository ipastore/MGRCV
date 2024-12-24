import numpy as np
import cv2
import matplotlib.pyplot as plt

path = '/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/utils/SuperGluePretrainedNetwork/colon_matches/f_0064556_f_0064636_matches.npz'
npz = np.load(path, allow_pickle=True)

print("Kepoints0 shape: ", npz['keypoints0'].shape)
print("Kepoints1 shape: ", npz['keypoints1'].shape)
print("Matches shape: ", npz['matches'].shape)
print("Number of matches: ", np.sum(npz['matches']>-1))
print("Match confidence shape: ", npz['match_confidence'].shape)
print("After RANSAC:\n")
print("inliers_keypoints0 shape: ", npz['inliers_keypoints0'].shape)
print("inliers_keypoints1 shape: ", npz['inliers_keypoints1'].shape)
print("inliers_confidence shape: ", npz['inliers_confidence'].shape)
print("fundamental_matrix shape: ", npz['fundamental_matrix'].shape)
print("Outliers keypoints0 shape: ", npz['outliers_keypoints0'].shape)
print("Outliers keypoints1 shape: ", npz['outliers_keypoints1'].shape)
print("Outliers confidence shape: ", npz['outliers_confidence'].shape)

#Load images png
img1 = cv2.imread('/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/data/Seq_027/f_0064556.png')
img2 = cv2.imread('/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/data/Seq_035/f_0082412.png')

# Resize images to original dimensions
img1 = cv2.resize(img1, (img1.shape[1], img1.shape[0]))
img2 = cv2.resize(img2, (img2.shape[1], img2.shape[0]))

# Print shapes
print("Image 1 shape: ", img1.shape)
print("Image 2 shape: ", img2.shape)

# Load inliners mask
inlier_mask = npz['inlier_mask']

# Plot matches, outliers and inliers
# Matches
matches = npz['matches']
match_confidence = npz['match_confidence']
# Convert to sequence of DMatch objects

# Keypoints
keypoints_0 = npz['keypoints0']
keypoints_1 = npz['keypoints1']

# Filter valid matches
valid_matches = matches > -1  # Boolean mask where matches_SG > -1 are valid matches
matched_keypoints0 = keypoints_0[valid_matches]  # x1 points from image 1
matched_keypoints1 = keypoints_1[matches[valid_matches]]  # x2 points from image 2

# Convert keypoints to cv2.KeyPoint
keypoints_0 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints_0]
keypoints_1 = [cv2.KeyPoint(x=float(kp[0]), y=float(kp[1]), size=1) for kp in keypoints_1]

# # Outliers
# mkpts0_outliers = npz['outliers_keypoints0']
# mkpts1_outliers = npz['outliers_keypoints1']
# mconf_outliers = npz['outliers_confidence']

# # Inliers
# mkpts0_inliers = npz['inliers_keypoints0']
# mkpts1_inliers = npz['inliers_keypoints1']
# mconf_inliers = npz['inliers_confidence']


# Create DMatch objects for valid matches
dMatchesList = [cv2.DMatch(_queryIdx=i, _trainIdx=matches[i], _distance=match_confidence[i])
                for i in range(len(matches)) if valid_matches[i]]

# Generate inliers and outliers for SuperGlue
inliers_matches = [
    dMatchesList[i] for i in range(len(dMatchesList)) if inlier_mask[i]
]
outliers_matches = [
    dMatchesList[i] for i in range(len(dMatchesList)) if not inlier_mask[i]
]

# Draw RANSAC inliers and outliers
imgMatched_RANSAC = cv2.drawMatches(img1, keypoints_0, img2, keypoints_1, dMatchesList,
                            None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

imgInliersMatched = cv2.drawMatches(img1, keypoints_0, img2, keypoints_1, inliers_matches, None,
                                    matchColor=(0, 255, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

imgOutliersMatched = cv2.drawMatches(img1, keypoints_0, img2, keypoints_1, outliers_matches, None,
                                    matchColor=(255, 0, 0), flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

# Convert BGR images to RGB for displaying with matplotlib
imgMatched_RANSAC = cv2.cvtColor(imgMatched_RANSAC, cv2.COLOR_BGR2RGB)
imgInliersMatched = cv2.cvtColor(imgInliersMatched, cv2.COLOR_BGR2RGB)
imgOutliersMatched = cv2.cvtColor(imgOutliersMatched, cv2.COLOR_BGR2RGB)

# # Show images
plt.figure(figsize=(20, 20))

# Plot all matches
plt.subplot(311)
plt.imshow(imgMatched_RANSAC)
plt.title("All matches")

# Plot inliers
plt.subplot(312)
plt.imshow(imgInliersMatched)
plt.title("Inliers")

# Plot outliers
plt.subplot(313)
plt.imshow(imgOutliersMatched)
plt.title("Outliers")

plt.show()