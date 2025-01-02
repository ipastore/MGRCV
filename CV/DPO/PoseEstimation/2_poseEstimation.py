import numpy as np
import sfm
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
import os

# ============================
# INITIAL SETUP AND PARAMETERS
# ============================

# File paths and image identifiers
REFERENCE_IMAGE = 'Img02_Try1_12M'
FIRST_IMAGE = 'Img25_Try1_12M'
AVAILABLE_IMAGES = ['Img14_Try1_12M']

# Construct the full path to the npz file
npz_file_path = os.path.join(
    os.path.dirname(__file__), 
    f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{FIRST_IMAGE}_inliers.npz'
)
data = np.load(npz_file_path)

# Keypoints and matches for initial pair
keypoints_reference = data['keypoints0']
keypoints_first = data['keypoints1']
matches_initial = data['matches']
inliers_mask_initial = data['inliers_matches']


# Extract matched inliers
inliers_reference = keypoints_reference[inliers_mask_initial[:, 0]]
inliers_first = keypoints_first[inliers_mask_initial[:, 1]]

# Convert to homogeneous coordinates
homogeneous_reference = np.vstack((inliers_reference.T, np.ones(inliers_reference.T.shape[1])))
homogeneous_first = np.vstack((inliers_first.T, np.ones(inliers_first.T.shape[1])))

# Image paths for visualization
ref_img_path = os.path.join(
    os.path.dirname(__file__), 
    f'../Images/Set_12MP/EntireSet/{REFERENCE_IMAGE}.jpg'
)
first_img_path = os.path.join(
    os.path.dirname(__file__), 
    f'../Images/Set_12MP/EntireSet/{FIRST_IMAGE}.jpg'
)
ref_img = plt.imread(ref_img_path)
first_img = plt.imread(first_img_path)

# ============================
# FUNDAMENTAL MATRIX AND ESSENTIAL MATRIX
# ============================

# Load fundamental matrix
fundamental_matrix_path = os.path.join(
    os.path.dirname(__file__), 
    f'../RANSAC/results/fundamental/F_{REFERENCE_IMAGE}_vs_{FIRST_IMAGE}.txt'
)
F_estimated = sfm.load_matrix(fundamental_matrix_path)
sfm.visualize_epipolar_lines(F_estimated, ref_img, first_img, show_epipoles=True, automatic=False)

# Load camera intrinsics
camera_intrinsics_path = os.path.join(
    os.path.dirname(__file__), 
    '../Camera_calibration/Calibration_12MP/K_Calibration_12MP.txt'
)
K_camera = sfm.load_matrix(camera_intrinsics_path)

# Compute essential matrix
essential_matrix = sfm.compute_essential_matrix_from_F(F_estimated, K_camera, K_camera)
print("Estimated Essential Matrix:")
print(essential_matrix)

# Decompose essential matrix
R1, R2, t_candidates = sfm.decompose_essential_matrix(essential_matrix)
R_correct, t_correct, points_3D_initial = sfm.select_correct_pose(
    homogeneous_reference, homogeneous_first, K_camera, K_camera, R1, R2, t_candidates
)

print("Correct Rotation (Relative):")
print(R_correct)
print("Correct Translation (Relative):")
print(t_correct)

# ============================
# INITIAL BUNDLE ADJUSTMENT
# ============================

# Projection matrices
P_ref_initial = sfm.get_projection_matrix(K_camera, np.eye(4))
T_first_to_ref_initial = sfm.ensamble_T(R_correct, t_correct)
P_first_initial = sfm.get_projection_matrix(K_camera, T_first_to_ref_initial)

# Project points into cameras
projected_ref_initial = sfm.project_to_camera(P_ref_initial, points_3D_initial)
projected_first_initial = sfm.project_to_camera(P_first_initial, points_3D_initial)

# Visualize initial residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, inliers_reference.T, projected_ref_initial, "Initial Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(first_img, inliers_first.T, projected_first_initial, "Initial Residuals in First Image", ax=axs[1])
plt.tight_layout()
plt.show()

# ============================
# OPTIMIZATION WITH BUNDLE ADJUSTMENT
# ============================

# Prepare data for optimization
initial_guess_theta = sfm.crossMatrixInv(logm(R_correct.astype('float64')))
t_norm = np.linalg.norm(t_correct)
t_theta = np.arccos(t_correct[2] / t_norm)
t_phi = np.arctan2(t_correct[1], t_correct[0])

initial_guess = np.hstack((
    initial_guess_theta, t_theta, t_phi, points_3D_initial[:3, :].flatten()
))
num_points = points_3D_initial.shape[1]

# Optimize
optimized_result = least_squares(
    sfm.resBundleProjection,
    initial_guess,
    args=(inliers_reference.T, inliers_first.T, K_camera, num_points),
    method='lm'
)

# Extract optimized parameters
optimized_theta = optimized_result.x[:3]
t_theta_optimized = optimized_result.x[3]
t_phi_optimized = optimized_result.x[4]
t_c2_c1_opt = np.array([
    np.sin(t_theta_optimized) * np.cos(t_phi_optimized),
    np.sin(t_theta_optimized) * np.sin(t_phi_optimized),
    np.cos(t_theta_optimized)
])
optimized_3D_points = optimized_result.x[5:].reshape(3, -1).T

# ============================
# VISUALIZE OPTIMIZED RESULTS
# ============================

# Reconstruct optimized poses
optimized_rotation = expm(sfm.crossMatrix(optimized_theta))
optimized_transformation = sfm.ensamble_T(optimized_rotation, t_c2_c1_opt)
P_first_optimized = sfm.get_projection_matrix(K_camera, optimized_transformation)

# Project optimized points
optimized_3D_points_h = np.vstack((optimized_3D_points.T, np.ones((1, optimized_3D_points.shape[0]))))
projected_ref_optimized = sfm.project_to_camera(P_ref_initial, optimized_3D_points_h)
projected_first_optimized = sfm.project_to_camera(P_first_optimized, optimized_3D_points_h)

# Visualize optimized residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, inliers_reference.T, projected_ref_optimized, "Optimized Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(first_img, inliers_first.T, projected_first_optimized, "Optimized Residuals in First Image", ax=axs[1])
plt.tight_layout()
plt.show()

# ============================
# PNP FOR ADDITIONAL CAMERAS
# ============================

# Construct the full path to the npz file
pnp_npz_file_path = os.path.join(
    os.path.dirname(__file__), 
    f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{AVAILABLE_IMAGES[0]}_inliers.npz'
)

# Keypoints and matches for initial pair
pnp_data = np.load(pnp_npz_file_path)
pnp_keypoints_ref = pnp_data['keypoints0']
pnp_keypoints_NewCamera = pnp_data['keypoints1']
pnp_inliers_mask = pnp_data['inliers_matches']
# Extract matched inliers
previous_inliers_indices = set(inliers_mask_initial[:, 0])
pnp_inliers_visible_mask = np.array([match for match in pnp_inliers_mask if match[0] in previous_inliers_indices])
pnp_KeyMatches_ref = pnp_keypoints_ref[pnp_inliers_visible_mask[:, 0]]
pnp_KeyMatches_NewCamera = pnp_keypoints_NewCamera[pnp_inliers_visible_mask[:, 1]]
print("Numer of matches inliers of the new camera:", pnp_inliers_mask.shape)
print("Numer of matches inliers of the new camera visible previously:", pnp_inliers_visible_mask.shape)

# Convert to homogeneous coordinates
x_PnP_h = np.vstack((pnp_KeyMatches_NewCamera.T, np.ones(pnp_KeyMatches_NewCamera.T.shape[1])))
imagePoints = np.ascontiguousarray(x_PnP_h[0:2, :].T).reshape((x_PnP_h.shape[1], 1, 2))
visible_3D_points = optimized_3D_points[pnp_inliers_visible_mask[:, 0]]
retval, theta_c3_c1_initial, t_c3_c1_initial = cv2.solvePnP(visible_3D_points, imagePoints, K_camera, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
R_c3_c1_initial = expm(sfm.crossMatrix(theta_c3_c1_initial))
print("t_c3_c1_initial:", t_c3_c1_initial)
print("theta_c3_c1_initial:", theta_c3_c1_initial)
print("Initial Rotation (Relative):", R_c3_c1_initial)




