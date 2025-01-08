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
# , 'Img12_Try2_12M'

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
# PROCESO AUTOMÁTICO MULTICÁMARA
# ============================

# Inicializar variables
rotations = [np.eye(3)]  # Rotación identidad para cámara de referencia
translations = [np.zeros(3)]  # Traslación fija para cámara de referencia
optimized_3DPoints = Cam2_3DPoints_opt  # Puntos 3D optimizados hasta ahora

for idx, target_image in enumerate(AVAILABLE_IMAGES, start=2):  # idx empieza en 2 porque Cam2 es la primera adicional
    print(f"Procesando cámara {idx} con imagen: {target_image}")

    # ============================
    # PNP Y TRIANGULACIÓN
    # ============================

    R_target, t_target, point3D_map, target_3DPoints = sfm.perform_pnp_and_triangulate(
        REFERENCE_IMAGE, target_image, K_camera, point3D_map, optimized_3DPoints, ref_cam_id=1, target_cam_id=idx
    )
    print(f"Rotación (Cámara {idx}):\n", R_target)
    print(f"Traslación (Cámara {idx}):\n", t_target)
    print(f"Número de puntos 3D tras PnP: {target_3DPoints.shape[0]}")

    # ================================
    # VISUALIZACIÓN INICIAL DE RESIDUALES
    # ================================

    target_img = sfm.get_image(target_image)
    T_target = sfm.ensamble_T(R_target, t_target)
    P_target = sfm.get_projection_matrix(K_camera, T_target)

    # Filtrar puntos visibles en ambas cámaras
    target_idx_visible_points = [
        pid for pid, cameras in point3D_map.items() if 1 in cameras and idx in cameras
    ]
    filtered_3DPoints_h = np.vstack((
        np.array([optimized_3DPoints[pid] for pid in target_idx_visible_points]).T,
        np.ones(len(target_idx_visible_points))
    ))

    # Cargar correspondencias
    data_pair = np.load(f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{target_image}_inliers.npz')
    target_keypoints = data_pair['keypoints1']
    mask = data_pair['inliers_matches']
    ref_2DPoints_pair = ref_keypoints[mask[:, 0]]
    target_2DPoints_pair = target_keypoints[mask[:, 1]]
    del data_pair

    # Proyecciones iniciales
    ref_proj2DPoints = sfm.project_to_camera(P_ref, filtered_3DPoints_h)
    target_proj2DPoints = sfm.project_to_camera(P_target, filtered_3DPoints_h)

    # Visualización
    ref_FilteredKeypoints = np.array([ref_keypoints[point3D_map[pid][1]] for pid in target_idx_visible_points])
    target_FilteredKeypoints = np.array([target_keypoints[point3D_map[pid][idx]] for pid in target_idx_visible_points])
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    sfm.visualize_residuals(ref_img, ref_FilteredKeypoints.T, ref_proj2DPoints, f"Residuals en Ref (Antes de opt)", ax=axs[0])
    sfm.visualize_residuals(target_img, target_FilteredKeypoints.T, target_proj2DPoints, f"Residuals en Cámara {idx} (Antes de opt)", ax=axs[1])
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # ============================
    # BUNDLE AJUSTMENT
    # ============================

    rotations.append(R_target)
    translations.append(t_target)

    # Crear keypoints_2D
    keypoints_2D = [ref_keypoints.T] + [
        np.load(f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{img}_inliers.npz')['keypoints1'].T for img in AVAILABLE_IMAGES[:idx - 1]
    ] + [target_keypoints.T]

    # Preparar el vector de parámetros
    op_vector = sfm.prepare_op_vector(rotations, translations, target_3DPoints.T)
    print(f"Vector de parámetros para optimización: {op_vector.shape}")

    # Ejecutar la optimización
    optimized_result = least_squares(
        sfm.resBundleProjection_multicamera,
        op_vector,
        args=(point3D_map, keypoints_2D, K_camera, len(keypoints_2D)),
        method='lm'
    )

    # Recuperar parámetros optimizados
    optimized_data = sfm.recover_parameters(optimized_result.x, len(keypoints_2D), target_3DPoints.shape[0])
    rotations = optimized_data["rotations"]
    translations = optimized_data["translations"]
    optimized_3DPoints = optimized_data["points_3D"]

    print(f"Cámara {idx} optimizada. Puntos 3D optimizados: {optimized_3DPoints.shape[0]}")

    # ============================
    # VISUALIZAR RESULTADOS OPTIMIZADOS
    # ============================

    R_target_opt = expm(sfm.crossMatrix(rotations[idx - 1]))
    T_target_opt = sfm.ensamble_T(R_target_opt, translations[idx - 1])
    P_target_opt = sfm.get_projection_matrix(K_camera, T_target_opt)

    ref_proj2DPoints_opt = sfm.project_to_camera(P_ref, filtered_3DPoints_h)
    target_proj2DPoints_opt = sfm.project_to_camera(P_target_opt, filtered_3DPoints_h)

    fig, axs = plt.subplots(2, 2, figsize=(18, 6))
    sfm.visualize_residuals(ref_img, ref_FilteredKeypoints.T, ref_proj2DPoints, f"Residuals en Ref (Antes de opt)", ax=axs[0, 0])
    sfm.visualize_residuals(ref_img, ref_FilteredKeypoints.T, ref_proj2DPoints_opt, f"Residuals en Ref (Después de opt)", ax=axs[0, 1])
    sfm.visualize_residuals(target_img, target_FilteredKeypoints.T, target_proj2DPoints, f"Residuals en Cámara {idx} (Antes de opt)", ax=axs[1, 0])
    sfm.visualize_residuals(target_img, target_FilteredKeypoints.T, target_proj2DPoints_opt, f"Residuals en Cámara {idx} (Después de opt)", ax=axs[1, 1])
    plt.tight_layout()
    plt.show()
    plt.close(fig)

print("PROCESO COMPLETADO.")