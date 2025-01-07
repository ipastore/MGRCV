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
AVAILABLE_IMAGES = ['Img15_Try3_12M']


data = sfm.load_npz_data('../RANSAC/results/inliers', REFERENCE_IMAGE, FIRST_IMAGE)
ref_keypoints = data['keypoints0']
cam2_keypoints = data['keypoints1']
print(ref_keypoints.shape, cam2_keypoints.shape)
matches_pair2 = data['matches']
mask2= data['inliers_matches']
del data

# Extract matched inliers
ref_2DPoints_pair2 = ref_keypoints[mask2[:, 0]]
cam2_2DPoints_pair2 = cam2_keypoints[mask2[:, 1]]

# Convert to homogeneous coordinates
ref_3DPoints_h_pair2 = np.vstack((ref_2DPoints_pair2.T, np.ones(ref_2DPoints_pair2.T.shape[1])))
cam2_3DPoints_h_pair2 = np.vstack((cam2_2DPoints_pair2.T, np.ones(cam2_2DPoints_pair2.T.shape[1])))

# Image paths for visualization
ref_img = sfm.get_image(REFERENCE_IMAGE)
cam2_img = sfm.get_image(FIRST_IMAGE)

# ============================
# FUNDAMENTAL MATRIX AND ESSENTIAL MATRIX
# ============================

#### FUNDAMENTAL MATRIX ####
# Load fundamental matrix amd Intrinsec camera data
fundamental_matrix_path = os.path.join(os.path.dirname(__file__), f'../RANSAC/results/fundamental/F_{REFERENCE_IMAGE}_vs_{FIRST_IMAGE}.txt')
camera_intrinsics_path = os.path.join(os.path.dirname(__file__), '../Camera_calibration/Calibration_12MP/K_Calibration_12MP.txt')
F_estimated = sfm.load_matrix(fundamental_matrix_path)
K_camera = sfm.load_matrix(camera_intrinsics_path)
sfm.visualize_epipolar_lines(F_estimated, ref_img, cam2_img, show_epipoles=True, automatic=False)

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
R_cam2, t_cam2, points_3D_initial = sfm.select_correct_pose(
    ref_3DPoints_h_pair2, cam2_3DPoints_h_pair2, K_camera, K_camera, R1, R2, t_candidates
)
print("Correct Rotation (Relative):")
print(R_cam2)
print("Correct Translation (Relative):")
print(t_cam2)
print("Magnitude of the translation vector:", np.linalg.norm(t_cam2))

# DICTIONARY:  3D POINTS - KEYPOINT MAPPING
point3D_map = {
    i: {1: mask2[i, 0], 2: mask2[i, 1]} for i in range(len(points_3D_initial.T))
}

# ================================
# VISUALIZATION OF INITIAL RESULTS
# ================================

# Projection matrices
P_ref = sfm.get_projection_matrix(K_camera, np.eye(4))
T_c_to_ref = sfm.ensamble_T(R_cam2, t_cam2)
P_cam2 = sfm.get_projection_matrix(K_camera, T_c_to_ref)

# Project points into cameras
ref_proj2DPoints_pair2 = sfm.project_to_camera(P_ref, points_3D_initial)
cam2_proj2DPoints_pair2 = sfm.project_to_camera(P_cam2, points_3D_initial)

# Visualize initial residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_2DPoints_pair2.T, ref_proj2DPoints_pair2, "Reference Image - Initial Residual", ax=axs[0])
sfm.visualize_residuals(cam2_img, cam2_2DPoints_pair2.T, cam2_proj2DPoints_pair2, "Initial Residuals in First Image", ax=axs[1])
plt.tight_layout()
plt.show()
plt.close(fig)


# ===========================================
# INITIAL OPTIMIZATION WITH BUNDLE ADJUSTMENT
# ===========================================

# Prepare data for optimization
initial_guess_theta_pair2 = sfm.crossMatrixInv(logm(R_cam2.astype('float64')))
t_norm_pair2 = np.linalg.norm(t_cam2)
t_theta_pair2 = np.arccos(t_cam2[2] / t_norm_pair2)
t_phi_pair2 = np.arctan2(t_cam2[1], t_cam2[0])
num_points = points_3D_initial.shape[1]

op_vector = np.hstack((
    initial_guess_theta_pair2,
    t_theta_pair2,
    t_phi_pair2,
    points_3D_initial[:3, :].flatten()
))

# Optimize
optimized_result = least_squares(
    sfm.resBundleProjection,
    op_vector,
    args=(ref_2DPoints_pair2.T, cam2_2DPoints_pair2.T, K_camera, num_points),
    method='lm'
)

# Extract optimized parameters
theta_c2_ref_opt = optimized_result.x[:3]
t_theta_optimized = optimized_result.x[3]
t_phi_optimized = optimized_result.x[4]
t_c2_opt = np.array([
    np.sin(t_theta_optimized) * np.cos(t_phi_optimized),
    np.sin(t_theta_optimized) * np.sin(t_phi_optimized),
    np.cos(t_theta_optimized)
])
Cam2_3DPoints_opt = optimized_result.x[5:].reshape(3, -1).T
print("Updated number of ·D points:", Cam2_3DPoints_opt.shape[0])

# ============================
# VISUALIZE OPTIMIZED RESULTS
# ============================

# Reconstruct optimized poses
R_Cam2_opt = expm(sfm.crossMatrix(theta_c2_ref_opt))
T_Cam2_opt = sfm.ensamble_T(R_Cam2_opt, t_c2_opt)
P_Cam2_opt = sfm.get_projection_matrix(K_camera, T_Cam2_opt)

# Project optimized points
Cam2_3DPoints_H_opt = np.vstack((Cam2_3DPoints_opt.T, np.ones((1, Cam2_3DPoints_opt.shape[0]))))
ref_proj2DPoints_pair2_opt = sfm.project_to_camera(P_ref, Cam2_3DPoints_H_opt)
Cam2_proj2DPoints_pair2_opt = sfm.project_to_camera(P_Cam2_opt, Cam2_3DPoints_H_opt)

# Visualize optimized residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_2DPoints_pair2.T, ref_proj2DPoints_pair2_opt, "Optimized Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(cam2_img, cam2_2DPoints_pair2.T, Cam2_proj2DPoints_pair2_opt, "Optimized Residuals in First Image", ax=axs[1])
plt.tight_layout()
plt.show()
plt.close(fig)

# ============================
# PNP FOR ADDITIONAL CAMERAS
# ============================

R_Cam3, t_Cam3, point3D_map, Cam3_3DPoints = sfm.perform_pnp_and_triangulate(
    REFERENCE_IMAGE, AVAILABLE_IMAGES[0], K_camera, point3D_map, Cam2_3DPoints_opt, ref_cam_id=1, target_cam_id=3
)

print("Rotation matrix (Camera 3 with respect to Camera 1):\n", R_Cam3)
print("Translation vector (Camera 3 with respect to Camera 1):\n", t_Cam3)
print("Number of 3D points after PnP and triangulation:", Cam3_3DPoints.shape[0])


# ================================
# VISUALIZATION OF INITIAL RESULTS
# ================================

cam3_img = sfm.get_image(AVAILABLE_IMAGES[0])

# Projection matrices
T_Cam3 = sfm.ensamble_T(R_Cam3, t_Cam3)
P_Cam3 = sfm.get_projection_matrix(K_camera, T_Cam3)

# Filtrar los puntos visibles en ambas cámaras
Cam3_idx_visible_points = [
    pid for pid, cameras in point3D_map.items() if 1 in cameras and 3 in cameras
]
# Proyección en coordenadas homogéneas para los puntos visibles
Cam3_filtered3DPoints_h = np.vstack((
    np.array([Cam3_3DPoints[pid] for pid in Cam3_idx_visible_points]).T,
    np.ones(len(Cam3_idx_visible_points))
))

# Cargar correspondencias entre las cámaras
data_pair3 = np.load(os.path.join(os.path.dirname(__file__),f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{AVAILABLE_IMAGES[0]}_inliers.npz'))
# pnp_keypoints_ref = data_pair3['keypoints0']
cam3_keypoints = data_pair3['keypoints1']
mask3 = data_pair3['inliers_matches']
# Extract matched inliers
ref_2DPoints_pair3 = ref_keypoints[mask3[:, 0]]
cam3_2DPoints_pair3 = cam3_keypoints[mask3[:, 1]]
del data_pair3

ref_proj2DPoints_pair3 = sfm.project_to_camera(P_ref, Cam3_filtered3DPoints_h)
cam3_proj2DPoints_pair3 = sfm.project_to_camera(P_Cam3, Cam3_filtered3DPoints_h)
# Obtener los keypoints correspondientes en ambas cámaras
ref_FilteredKeypoints_pair3 = np.array([ref_keypoints[point3D_map[pid][1]] for pid in Cam3_idx_visible_points])
cam3_FilteredKeypoints_pair3 = np.array([cam3_keypoints[point3D_map[pid][3]] for pid in Cam3_idx_visible_points])

# Visualize initial residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_FilteredKeypoints_pair3.T, ref_proj2DPoints_pair3, "Initial Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(cam3_img, cam3_FilteredKeypoints_pair3.T, cam3_proj2DPoints_pair3, "Initial Residuals in PnP Image", ax=axs[1])
plt.tight_layout()
plt.show()
plt.close(fig)

# ============================
# BUNDLE AJUSTMENT FOR THE 3RD CAMERA
# ============================

R_list = [R_cam2, R_Cam3]
t_list = [t_cam2, t_Cam3]

# Transformar al formato esperado
keypoints_2D = [
    ref_keypoints.T,
    cam2_keypoints.T,
    cam3_keypoints.T   
]

# Preparar el vector de parámetros
op_vector = sfm.prepare_op_vector(R_list, t_list, Cam3_3DPoints.T)
print(f"Vector de parámetros para optimización: {op_vector.shape}")


# Ejecutar la optimización
optimized_result = least_squares(
    sfm.resBundleProjection_multicamera,
    op_vector,
    args=(point3D_map, keypoints_2D, K_camera, len(keypoints_2D)),
    method='lm'  # Levenberg-Marquardt (puedes ajustar según tu problema)
)

# Datos de entrada
n_cameras = 3  # Por ejemplo, 3 cámaras (ref, cam2, cam3)
n_points = Cam3_3DPoints.shape[0]  # Por ejemplo, 5073 puntos 3D

# Recuperar parámetros optimizados
optimized_data = sfm.recover_parameters(optimized_result.x, n_cameras, n_points)

# Acceder a los datos recuperados
rotations = optimized_data["rotations"]  # Lista de rotaciones (logm)
translations = optimized_data["translations"]  # Lista de traslaciones (3,)
Cam3_3DPoints_opt = optimized_data["points_3D"]  # Puntos 3D optimizados (n_points x 3)

# Ejemplo: Imprimir resultados
print("Rotaciones optimizadas (logm):", rotations)
print("Traslaciones optimizadas:", translations)
print("Puntos 3D optimizados (shape):", Cam3_3DPoints_opt.shape)
print("Magnitude of the translation vector:", np.linalg.norm(translations[1]))

# ============================
# VISUALIZE OPTIMIZED RESULTS
# ============================

# Reconstruct optimized poses
R_Cam3_opt = expm(sfm.crossMatrix(rotations[2]))
T_Cam3_to_ref_opt = sfm.ensamble_T(R_Cam3_opt, translations[2])
P_Cam3_opt = sfm.get_projection_matrix(K_camera, T_Cam3_to_ref_opt)

# Proyección en coordenadas homogéneas para los puntos visibles
Cam3_filtered3DPoints_h_opt = np.vstack((
    np.array([Cam3_3DPoints_opt[pid] for pid in Cam3_idx_visible_points]).T,
    np.ones(len(Cam3_idx_visible_points))
))

ref_proj2DPoints_pair3_opt = sfm.project_to_camera(P_ref, Cam3_filtered3DPoints_h_opt)
cam3_proj2DPoints_pair3_opt = sfm.project_to_camera(P_Cam3_opt, Cam3_filtered3DPoints_h_opt)

# Visualize optimized residuals
fig, axs = plt.subplots(2, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_FilteredKeypoints_pair3.T, ref_proj2DPoints_pair3, "Reference Image BEFORE opt", ax=axs[0, 0])
sfm.visualize_residuals(ref_img, ref_FilteredKeypoints_pair3.T, ref_proj2DPoints_pair3_opt, "Reference Image AFTER opt", ax=axs[0, 1])
sfm.visualize_residuals(cam3_img, cam3_FilteredKeypoints_pair3.T, cam3_proj2DPoints_pair3, "Cam3 BEFORE opt", ax=axs[1, 0])
sfm.visualize_residuals(cam3_img, cam3_FilteredKeypoints_pair3.T, cam3_proj2DPoints_pair3_opt, "Cam3 AFTER opt", ax=axs[1, 1])
plt.tight_layout()
plt.show()
plt.close(fig)

print("FINISIHED")


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sfm.drawRefSystem(ax, np.eye(4), '-', 'C1')
sfm.drawRefSystem(ax, T_Cam2_opt, '-', 'C2')
sfm.drawRefSystem(ax, np.linalg.inv(T_Cam3_to_ref_opt), '-', 'C3')
ax.scatter(Cam3_3DPoints_opt.T[0, :], Cam3_3DPoints_opt.T[1, :], Cam3_3DPoints_opt.T[2, :], marker='o', color='g', label='3D Points')
xFakeBoundingBox = np.linspace(-6, 10, 2)
yFakeBoundingBox = np.linspace(-6, 10, 2)
zFakeBoundingBox = np.linspace(-6, 10, 2)
plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
plt.show()
plt.close(fig)