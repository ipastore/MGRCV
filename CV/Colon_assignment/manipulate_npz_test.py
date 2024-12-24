import numpy as np
import cv2

# path = '/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/utils/SuperGluePretrainedNetwork/colon_matches/f_0064556_f_0064636_matches.npz'
# npz = np.load(path, allow_pickle=True)

# print("Kepoints0 shape: ", npz['keypoints0'].shape)
# print("Kepoints1 shape: ", npz['keypoints1'].shape)
# print("Matches shape: ", npz['matches'].shape)
# print("Number of matches: ", np.sum(npz['matches']>-1))
# print("Match confidence shape: ", npz['match_confidence'].shape)

#Load images png
img1 = cv2.imread('/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/data/Seq_027/f_0064556.png')
img2 = cv2.imread('/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/data/Seq_035/f_0082412.png')

# Print shapes
print("Image 1 shape: ", img1.shape)
print("Image 2 shape: ", img2.shape)
