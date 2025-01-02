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

# Function to load npz data
def load_npz_data(base_path, reference_image, target_image):
    npz_path = os.path.join(
        os.path.dirname(__file__), 
        f'{base_path}/{reference_image}_vs_{target_image}_inliers.npz'
    )
    return np.load(npz_path)

data = load_npz_data('../RANSAC/results/inliers', REFERENCE_IMAGE, FIRST_IMAGE)

# Keypoints and matches for initial pair
keypoints_reference = data['keypoints0']
keypoints_first = data['keypoints1']
matches_initial = data['matches']
inliers_mask_initial = data['inliers_matches']

# Extract matched inliers
inliers_reference = keypoints_reference[inliers_mask_initial[:, 0]]
inliers_first = keypoints_first[inliers_mask_initial[:, 1]]
idx_ref_global_points = inliers_mask_initial[:, 0]

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

#### FUNDAMENTAL MATRIX ####
# Load fundamental matrix amd Intrinsec camera data
fundamental_matrix_path = os.path.join(os.path.dirname(__file__), f'../RANSAC/results/fundamental/F_{REFERENCE_IMAGE}_vs_{FIRST_IMAGE}.txt')
camera_intrinsics_path = os.path.join(os.path.dirname(__file__), '../Camera_calibration/Calibration_12MP/K_Calibration_12MP.txt')
F_estimated = sfm.load_matrix(fundamental_matrix_path)
K_camera = sfm.load_matrix(camera_intrinsics_path)
sfm.visualize_epipolar_lines(F_estimated, ref_img, first_img, show_epipoles=True, automatic=False)

#### ESSENTIAL MATRIX ####
# Compute essential matrix
essential_matrix = sfm.compute_essential_matrix_from_F(F_estimated, K_camera, K_camera)
print("Estimated Essential Matrix:")
print(essential_matrix)


# ============================
# POSE ESTIMATION USING SFM
# ============================

# Decompose essential matrix
R1, R2, t_candidates = sfm.decompose_essential_matrix(essential_matrix)
R_correct, t_correct, points_3D_initial = sfm.select_correct_pose(
    homogeneous_reference, homogeneous_first, K_camera, K_camera, R1, R2, t_candidates
)

print("Correct Rotation (Relative):")
print(R_correct)
print("Correct Translation (Relative):")
print(t_correct)


# ================================
# VISUALIZATION OF INITIAL RESULTS
# ================================

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


# ===========================================
# INITIAL OPTIMIZATION WITH BUNDLE ADJUSTMENT
# ===========================================

# Prepare data for optimization
initial_guess_theta = sfm.crossMatrixInv(logm(R_correct.astype('float64')))
t_norm = np.linalg.norm(t_correct)
t_theta = np.arccos(t_correct[2] / t_norm)
t_phi = np.arctan2(t_correct[1], t_correct[0])
num_points = points_3D_initial.shape[1]

op_vector = np.hstack((
    initial_guess_theta,
    t_theta,
    t_phi,
    points_3D_initial[:3, :].flatten()
))

# Optimize
optimized_result = least_squares(
    sfm.resBundleProjection,
    op_vector,
    args=(inliers_reference.T, inliers_first.T, K_camera, num_points),
    method='lm'
)

# Extract optimized parameters
theta_c2_c1_opt = optimized_result.x[:3]
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
opt_rotation = expm(sfm.crossMatrix(theta_c2_c1_opt))
opt_transformation = sfm.ensamble_T(opt_rotation, t_c2_c1_opt)
P_first_optimized = sfm.get_projection_matrix(K_camera, opt_transformation)

# Project optimized points
optimized_3D_points_h = np.vstack((optimized_3D_points.T, np.ones((1, optimized_3D_points.shape[0]))))
projected_ref_optimized = sfm.project_to_camera(P_ref_initial, optimized_3D_points_h)
projected_first_optimized = sfm.project_to_camera(P_first_optimized, optimized_3D_points_h)

# Visualize optimized residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, inliers_reference.T, projected_ref_optimized, "Optimized Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(first_img, inliers_first.T, projected_first_optimized, "Optimized Residuals in First Image", ax=axs[1])
plt.tight_layout()

# ============================
# PNP FOR ADDITIONAL CAMERAS
# ============================

# Perform PnP for a new camera
def perform_pnp(reference_image, target_image, K_C, optimized_3D_points, inliers_mask_initial, idx_ref_global_points):
    npz_file_path = os.path.join(
        os.path.dirname(__file__), 
        f'../RANSAC/results/inliers/{reference_image}_vs_{target_image}_inliers.npz'
    )
    pnp_data = np.load(npz_file_path)
    pnp_keypoints_ref = pnp_data['keypoints0']
    pnp_keypoints_new = pnp_data['keypoints1']
    pnp_inliers_mask = pnp_data['inliers_matches']

    previous_inliers_indices = set(inliers_mask_initial[:, 0])
    pnp_inliers_visible_mask = np.array([
        match for match in pnp_inliers_mask if match[0] in previous_inliers_indices
    ])

    pnp_key_matches_ref = pnp_keypoints_ref[pnp_inliers_visible_mask[:, 0]]
    pnp_key_matches_new = pnp_keypoints_new[pnp_inliers_visible_mask[:, 1]]
    
    visible_3D_indices = [
        np.where(idx_ref_global_points == idx)[0][0] for idx in pnp_inliers_visible_mask[:, 0]
    ]
    filtered_3D_points = optimized_3D_points[visible_3D_indices]

    x_pnp_h = np.vstack((pnp_key_matches_new.T, np.ones(pnp_key_matches_new.T.shape[1])))
    image_points = np.ascontiguousarray(x_pnp_h[0:2, :].T).reshape((x_pnp_h.shape[1], 1, 2))
    # TODO: Comprobar que entran bien en formato los puntos 3d y 2d
    retval, rotation_vector, translation_vector = cv2.solvePnP(
        filtered_3D_points, image_points, K_C, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP
    )
    rotation_matrix = expm(sfm.crossMatrix(rotation_vector))
    return rotation_matrix, translation_vector, filtered_3D_points, pnp_key_matches_ref, pnp_key_matches_new

R_pnp, t_pnp, filtered_3D_points, pnp_key_matches_ref, pnp_key_matches_new = perform_pnp(
    REFERENCE_IMAGE, AVAILABLE_IMAGES[0], K_camera, optimized_3D_points, inliers_mask_initial, idx_ref_global_points
)

print("Rotation matrix (Camera 3 with respect to Camera 1):\n", R_pnp)
print("Translation vector (Camera 3 with respect to Camera 1):\n", t_pnp)


# ================================
# VISUALIZATION OF INITIAL RESULTS
# ================================

pnp_img_path = os.path.join(
    os.path.dirname(__file__), 
    f'../Images/Set_12MP/EntireSet/{AVAILABLE_IMAGES[0]}.jpg'
)

pnp_img = plt.imread(pnp_img_path)

# Projection matrices
T_pnp_to_ref_initial = sfm.ensamble_T(R_pnp, t_pnp)
P_pnp_initial = sfm.get_projection_matrix(K_camera, T_pnp_to_ref_initial)

# Project points into cameras
filtered_3D_points_h = np.vstack((filtered_3D_points.T, np.ones((1, filtered_3D_points.shape[0]))))
projected_ref_initial = sfm.project_to_camera(P_ref_initial, filtered_3D_points_h)
projected_pnp_initial = sfm.project_to_camera(P_pnp_initial, filtered_3D_points_h)

# Visualize initial residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, pnp_key_matches_ref.T, projected_ref_initial, "Initial Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(pnp_img, pnp_key_matches_new.T, projected_pnp_initial, "Initial Residuals in PnP Image", ax=axs[1])
plt.tight_layout()
plt.show()

# ============================
# BUNDLE AJUSTMENT FOR THE 3RD CAMERA
# ============================


