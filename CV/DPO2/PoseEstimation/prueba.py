import numpy as np
import sfm
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
import os
import time

# ============================
# INITIAL SETUP AND PARAMETERS
# ============================

# File paths and image identifiers
REFERENCE_IMAGE = 'Img02_Try1_12M'
FIRST_IMAGE = 'Img25_Try1_12M'

# REFERENCE_IMAGE = 'Img25_Try1_12M'
# FIRST_IMAGE = 'Img02_Try1_12M'
AVAILABLE_IMAGES = ['Img15_Try3_12M','Img23_Try1_12M']
# ,'Img23_Try1_12M','Img24_Try1_12M', 'Img14_Try1_12M', 'Img13_Try1_12M'

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

# RMSE antes del Bundle Adjustment
rmse_before = sfm.compute_rmse(cam2_2DPoints_pair2.T, cam2_proj2DPoints_pair2[:2,:])
print(f"RMSE antes del BA: {rmse_before:.4f}")

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
start_time = time.time()
optimized_result = least_squares(
    sfm.resBundleProjection,
    op_vector,
    args=(ref_2DPoints_pair2.T, cam2_2DPoints_pair2.T, K_camera, num_points),
    method='lm'
)
end_time = time.time()
elapsed_time = end_time - start_time
print(f"Tiempo de optimización Ref-Cam2: {elapsed_time:.6f} segundos")

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
print("Updated number of 3D points:", Cam2_3DPoints_opt.shape[0])

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

# RMSE después del Bundle Adjustment
rmse_after = sfm.compute_rmse(cam2_2DPoints_pair2.T, Cam2_proj2DPoints_pair2_opt[:2,:])
print(f"RMSE después del BA: {rmse_after:.4f}")

# Cambio porcentual en el RMSE
rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
print(f"Mejora porcentual en el RMSE: {rmse_improvement:.2f}%")

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
# Filtrar los puntos visibles en ambas cámaras
Cam2_idx_visible_points = [
    pid for pid, cameras in point3D_map.items() if 1 in cameras and 2 in cameras
]

R_list = [R_Cam2_opt] 
t_list = [t_c2_opt]
Img_list = []
keypoints_2D = [ref_keypoints.T,cam2_keypoints.T]
FilteredKeypoints_by_Pairs =[ref_2DPoints_pair2.T, cam2_2DPoints_pair2.T]
proj2DPoints_by_Pairs = [ref_proj2DPoints_pair2_opt[:2,:], Cam2_proj2DPoints_pair2_opt[:2,:]]
idx_visible_points_by_Pairs = [Cam2_idx_visible_points]


current_3DPoints_opt = Cam2_3DPoints_opt

for pnp_cam_idx in range(len(AVAILABLE_IMAGES)):

    pnp_camera_id = pnp_cam_idx + 3
    
    R_CamPNP, t_CamPNP, point3D_map, CamPNP_3DPoints = sfm.perform_pnp_and_triangulate(
        REFERENCE_IMAGE, AVAILABLE_IMAGES[pnp_cam_idx], K_camera, point3D_map, current_3DPoints_opt, ref_cam_id=1, target_cam_id=pnp_camera_id
    )
    
    # print(f"Rotation matrix (Camera {pnp_camera_id} with respect to Camera 1):\n", R_CamPNP)
    # print(f"Translation vector (Camera {pnp_camera_id} with respect to Camera 1):\n", t_CamPNP)
    print("Number of 3D points after PnP and triangulation:", CamPNP_3DPoints.shape[0])


    # ================================
    # VISUALIZATION OF INITIAL RESULTS
    # ================================

    camPnP_img = sfm.get_image(AVAILABLE_IMAGES[pnp_cam_idx])
    Img_list.append(camPnP_img)

    # Projection matrices
    T_CamPnP = sfm.ensamble_T(R_CamPNP, t_CamPNP)
    P_CamPnP = sfm.get_projection_matrix(K_camera, T_CamPnP)

    # Filtrar los puntos visibles en ambas cámaras
    CamPnP_idx_visible_points = [
        pid for pid, cameras in point3D_map.items() if 1 in cameras and pnp_camera_id in cameras
    ]
    
    # Proyección en coordenadas homogéneas para los puntos visibles
    CamPnP_filtered3DPoints_h = np.vstack((
        np.array([CamPNP_3DPoints[pid] for pid in CamPnP_idx_visible_points]).T,
        np.ones(len(CamPnP_idx_visible_points))
    ))
    
    # Cargar correspondencias entre las cámaras
    data_pairPnP = np.load(os.path.join(os.path.dirname(__file__),f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{AVAILABLE_IMAGES[pnp_cam_idx]}_inliers.npz'))
    # pnp_keypoints_ref = data_pair3['keypoints0']
    camPnP_keypoints = data_pairPnP['keypoints1']
    maskPnP = data_pairPnP['inliers_matches']
    # Extract matched inliers
    ref_2DPoints_pairPnP = ref_keypoints[maskPnP[:, 0]]
    camPnP_2DPoints_pairPnP = camPnP_keypoints[maskPnP[:, 1]]
    del data_pairPnP

    ref_proj2DPoints_pairPnP = sfm.project_to_camera(P_ref, CamPnP_filtered3DPoints_h)
    camPnP_proj2DPoints_pairPnP = sfm.project_to_camera(P_CamPnP, CamPnP_filtered3DPoints_h)
    # Obtener los keypoints correspondientes en ambas cámaras
    ref_FilteredKeypoints_pairPnP = np.array([ref_keypoints[point3D_map[pid][1]] for pid in CamPnP_idx_visible_points])
    camPnP_FilteredKeypoints_pairPnP = np.array([camPnP_keypoints[point3D_map[pid][pnp_camera_id]] for pid in CamPnP_idx_visible_points])

    # Visualize initial residuals
    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    sfm.visualize_residuals(ref_img, ref_FilteredKeypoints_pairPnP.T, ref_proj2DPoints_pairPnP, "Initial Residuals in Reference Image", ax=axs[0])
    sfm.visualize_residuals(camPnP_img, camPnP_FilteredKeypoints_pairPnP.T, camPnP_proj2DPoints_pairPnP, "Initial Residuals in PnP Image", ax=axs[1])
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    # ============================
    # BUNDLE AJUSTMENT FOR THE 3RD CAMERA
    # ============================
    
    # RMSE antes del Bundle Adjustment
    rmse_before = sfm.compute_rmse(camPnP_FilteredKeypoints_pairPnP.T, camPnP_proj2DPoints_pairPnP[:2,:])
    print(f"RMSE antes del BA: {rmse_before:.4f}")

    
    # Transformar al formato esperado
    R_list.append(R_CamPNP)
    t_list.append(t_CamPNP)
    keypoints_2D.append(camPnP_keypoints.T)
    n_cameras = pnp_camera_id  # Por ejemplo, 3 cámaras (ref, cam2, cam3)
    n_points = CamPNP_3DPoints.shape[0]  # Por ejemplo, 5073 puntos 3D

    # Preparar el vector de parámetros
    op_vector = sfm.prepare_op_vector(R_list, t_list, CamPNP_3DPoints.T)
    print(f"Vector de parámetros para optimización: {op_vector.shape}")

    # Ejecutar la optimización
    start_time = time.time()
    optimized_result = least_squares(
        sfm.resBundleProjection_multicamera,
        op_vector,
        args=(point3D_map, keypoints_2D, K_camera, n_cameras),
        method='trf', #'lm'  # Levenberg-Marquardt (puedes ajustar según tu problema)
        verbose=2
    )
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"Tiempo de optimización Ref-Cam{pnp_camera_id}: {elapsed_time:.6f} segundos")

    # Recuperar parámetros optimizados
    optimized_data = sfm.recover_parameters(optimized_result.x, n_cameras, n_points)

    # Acceder a los datos recuperados
    R_list = optimized_data["rotations"]  # Lista de rotaciones (logm)
    t_list = optimized_data["translations"]  # Lista de traslaciones (3,)
    current_3DPoints_opt = optimized_data["points_3D"]  # Puntos 3D optimizados (n_points x 3)

    # Ejemplo: Imprimir resultados
    # print("Rotaciones optimizadas (logm):", rotations)
    # print("Traslaciones optimizadas:", translations)
    # print("Puntos 3D optimizados (shape):", Cam3_3DPoints_opt.shape)
    print("Magnitude of the translation vector:", np.linalg.norm(t_list[0]))

    # ============================
    # VISUALIZE OPTIMIZED RESULTS
    # ============================

    # Reconstruct optimized poses
    R_CamPnP_opt = R_list[n_cameras-2]
    T_CamPnP_to_ref_opt = sfm.ensamble_T(R_CamPnP_opt, t_list[n_cameras-2])
    P_CamPnP_opt = sfm.get_projection_matrix(K_camera, T_CamPnP_to_ref_opt)

    # Proyección en coordenadas homogéneas para los puntos visibles
    CamPnP_filtered3DPoints_h_opt = np.vstack((
        np.array([current_3DPoints_opt[pid] for pid in CamPnP_idx_visible_points]).T,
        np.ones(len(CamPnP_idx_visible_points))
    ))

    ref_proj2DPoints_pairPnP_opt = sfm.project_to_camera(P_ref, CamPnP_filtered3DPoints_h_opt)
    camPnP_proj2DPoints_pairPnP_opt = sfm.project_to_camera(P_CamPnP_opt, CamPnP_filtered3DPoints_h_opt)
    
    # RMSE después del Bundle Adjustment
    rmse_after = sfm.compute_rmse(camPnP_FilteredKeypoints_pairPnP.T, camPnP_proj2DPoints_pairPnP_opt[:2,:])
    print(f"RMSE después del BA: {rmse_after:.4f}")
    
    # Cambio porcentual en el RMSE
    rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
    print(f"Mejora porcentual en el RMSE: {rmse_improvement:.2f}%")
    FilteredKeypoints_by_Pairs.append(camPnP_FilteredKeypoints_pairPnP.T)
    proj2DPoints_by_Pairs.append(camPnP_proj2DPoints_pairPnP_opt[:2,:])
    idx_visible_points_by_Pairs.append(CamPnP_idx_visible_points)
    
    # All the RMSEs values
    for idx in range(len(FilteredKeypoints_by_Pairs)):
        if idx == 0:
            P = P_ref
                # Filtrar los puntos visibles en ambas cámaras
            visible_points = [
                pid for pid, cameras in point3D_map.items() if 1 in cameras and (idx+2) in cameras
            ]
            filtered3DPoints_h_opt = np.vstack((
                np.array([current_3DPoints_opt[pid] for pid in visible_points]).T,
                np.ones(len(visible_points))
            ))
        else:
            R = R_list[idx-1]
            T = sfm.ensamble_T(R, t_list[idx-1])
            P = sfm.get_projection_matrix(K_camera, T)
            filtered3DPoints_h_opt = np.vstack((
                np.array([current_3DPoints_opt[pid] for pid in idx_visible_points_by_Pairs[idx-1]]).T,
                np.ones(len(idx_visible_points_by_Pairs[idx-1]))
            ))
        proj2DPoints_by_Pairs[idx] = sfm.project_to_camera(P, filtered3DPoints_h_opt)
        rmse_current = sfm.compute_rmse(FilteredKeypoints_by_Pairs[idx], proj2DPoints_by_Pairs[idx][:2,:])
        print(f"RMSE actual en la cámara {idx+1}: {rmse_current:.4f}")

    # Visualize optimized residuals
    fig, axs = plt.subplots(2, 2, figsize=(18, 6))
    sfm.visualize_residuals(ref_img, ref_FilteredKeypoints_pairPnP.T, ref_proj2DPoints_pairPnP, "Reference Image BEFORE opt", ax=axs[0, 0])
    sfm.visualize_residuals(ref_img, ref_FilteredKeypoints_pairPnP.T, ref_proj2DPoints_pairPnP_opt, "Reference Image AFTER opt", ax=axs[0, 1])
    sfm.visualize_residuals(camPnP_img, camPnP_FilteredKeypoints_pairPnP.T, camPnP_proj2DPoints_pairPnP, "Cam3 BEFORE opt", ax=axs[1, 0])
    sfm.visualize_residuals(camPnP_img, camPnP_FilteredKeypoints_pairPnP.T, camPnP_proj2DPoints_pairPnP_opt, "Cam3 AFTER opt", ax=axs[1, 1])
    plt.tight_layout()
    plt.show()
    plt.close(fig)

    print("FINISIHED")

# ============================
# VISUALIZE 3D RECONSTRUCTION
# ============================
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sfm.drawRefSystem(ax, np.eye(4), '-', 'C1_ref')
for extra_cameras_idx in range(len(R_list)):
    sfm.drawRefSystem(ax, np.linalg.inv(sfm.ensamble_T(R_list[extra_cameras_idx], t_list[extra_cameras_idx])), '-', f'C{extra_cameras_idx+2}')
ax.scatter(current_3DPoints_opt.T[0, :], current_3DPoints_opt.T[1, :], current_3DPoints_opt.T[2, :], marker='o', color='g', label='3D Points')
xFakeBoundingBox = np.linspace(-6, 10, 2)
yFakeBoundingBox = np.linspace(-6, 10, 2)
zFakeBoundingBox = np.linspace(-6, 10, 2)
plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
plt.show()
plt.close(fig)

# ============================
# SAVE RESULTS TO FILE
# ============================

# # Define the output file paths
# output_dir = os.path.join(os.path.dirname(__file__), 'results')
# os.makedirs(output_dir, exist_ok=True)
# R_file_path = os.path.join(output_dir, 'R_matrices.npy')
# T_file_path = os.path.join(output_dir, 'T_vectors.npy')
# points_3D_file_path = os.path.join(output_dir, '3D_points.npy')

# # Save the R matrices, T vectors, and 3D points
# sfm.save_transformations(R_list, t_list, "rotations.npz", "translations.npz")
# np.save(points_3D_file_path, current_3DPoints_opt)
# print(f"3D points saved to {points_3D_file_path}")



# # ===================================
# # DLT FOR POSE ESTIMATION OLD CAMERA
# # ===================================

camOld_img = sfm.get_image('Img00_Try1_12M')
dlt_data = np.load(os.path.join(os.path.dirname(__file__),f'../RANSAC/results/inliers/Img00_Try1_12M_vs_Img02_Try1_12M_inliers.npz'))
dlt_keypoints_ref = dlt_data['keypoints1']
dlt_keypoints_old = dlt_data['keypoints0']
dlt_mask = dlt_data['inliers_matches']
# Filtrar puntos existentes y actualizar `point3D_map`
for match in dlt_mask:
    ref_idx, old_idx = match[1], match[0]
    found = False
    for pid, cameras in point3D_map.items():
        if cameras.get(1) == ref_idx:  # Si el punto ya está en el diccionario
            point3D_map[pid][0] = old_idx  # Actualizar keypoint en cam3
            found = True
            break
        
# Filtrar los puntos visibles en ambas cámaras
CamOld_idx_visible_points = [
    pid for pid, cameras in point3D_map.items() if 1 in cameras and 0 in cameras
]

# Proyección en coordenadas homogéneas para los puntos visibles
CamOld_filtered3DPoints_h = np.vstack((
    np.array([current_3DPoints_opt[pid] for pid in CamOld_idx_visible_points]).T,
    np.ones(len(CamOld_idx_visible_points))
))

camOld_FilteredKeypoints_pairOld = np.array([dlt_keypoints_old[point3D_map[pid][0]] for pid in CamOld_idx_visible_points])
camOld_FilteredKeypoints_pairOld_h = np.vstack((camOld_FilteredKeypoints_pairOld.T, np.ones((1, camOld_FilteredKeypoints_pairOld.shape[0]))))

P_old, inliers_old = sfm.ransac_dlt(CamOld_filtered3DPoints_h.T, camOld_FilteredKeypoints_pairOld_h.T, threshold=5, max_iterations=1000)
print("Matriz de proyección P:")
print(P_old)
print("Inliers encontrados:")
print(np.where(inliers_old)[0])

inlier_indices = np.where(inliers_old)[0]
prueba_3d_points = CamOld_filtered3DPoints_h.T[inlier_indices]
prueba_2d_points = camOld_FilteredKeypoints_pairOld_h.T[inlier_indices]
ref_keypoints_filtered = np.array([dlt_keypoints_ref[point3D_map[pid][1]] for pid in CamOld_idx_visible_points])
ref_keypoints_filtered = ref_keypoints_filtered[inlier_indices]

ref_proj2DPoints_pairOld = sfm.project_to_camera(P_ref, prueba_3d_points.T)
camOld_proj2DPoints_pairOld = sfm.project_to_camera(P_old, prueba_3d_points.T)

# RMSE antes del Bundle Adjustment
rmse_before = sfm.compute_rmse(prueba_2d_points.T[:2,:], camOld_proj2DPoints_pairOld[:2,:])
print(f"RMSE antes del BA: {rmse_before:.4f}")

# Visualize initial residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_keypoints_filtered.T, ref_proj2DPoints_pairOld, "Initial Residuals in PnP Image", ax=axs[0])
sfm.visualize_residuals(camOld_img, prueba_2d_points.T, camOld_proj2DPoints_pairOld, "Initial Residuals in PnP Image", ax=axs[1])
plt.tight_layout()
plt.show()

K_old, R_old, t_old = sfm.decompose_projection_matrix_with_sign(P_old)
t_old = t_old[0:3]
# Normalizar K
K_old /= K_old[2, 2]
print("Intrinsics matrix K:")
print(K_old)
print("Rotation matrix R:")
print(R_old)
print("Translation vector t:")
print(t_old)

R_old_opt, t_old_opt = sfm.bundle_adjustment_old(prueba_3d_points, prueba_2d_points, K_old, R_old, t_old)

T_CamOld_opt = sfm.ensamble_T(R_old_opt, t_old_opt)
P_CamOld_opt = sfm.get_projection_matrix(K_old, T_CamOld_opt)
camOld_proj2DPoints_pairOld_opt = sfm.project_to_camera(P_CamOld_opt, prueba_3d_points.T)

# RMSE después del Bundle Adjustment
rmse_after = sfm.compute_rmse(prueba_2d_points.T[:2,:], camOld_proj2DPoints_pairOld_opt[:2,:])
print(f"RMSE después del BA: {rmse_after:.4f}")

# Cambio porcentual en el RMSE
rmse_improvement = (rmse_before - rmse_after) / rmse_before * 100
print(f"Mejora porcentual en el RMSE: {rmse_improvement:.2f}%")


# Visualize optimized residuals
fig, axs = plt.subplots(2, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_keypoints_filtered.T, ref_proj2DPoints_pairOld, "Reference Image BEFORE opt", ax=axs[0, 0])
sfm.visualize_residuals(ref_img, ref_keypoints_filtered.T, ref_proj2DPoints_pairOld, "Reference Image AFTER opt", ax=axs[0, 1])
sfm.visualize_residuals(camOld_img, prueba_2d_points.T, camOld_proj2DPoints_pairOld, "Cam3 BEFORE opt", ax=axs[1, 0])
sfm.visualize_residuals(camOld_img, prueba_2d_points.T, camOld_proj2DPoints_pairOld_opt, "Cam3 AFTER opt", ax=axs[1, 1])
plt.tight_layout()
plt.show()
plt.close(fig)


fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
sfm.drawRefSystem(ax, np.eye(4), '-', 'C1_ref')
# sfm.drawRefSystem(ax, (sfm.ensamble_T(R_old_opt, t_old_opt)), '-', 'C_OLD')
sfm.drawRefSystem(ax, np.linalg.inv(sfm.ensamble_T(R_old_opt, t_old_opt)), '-', 'C_OLD')
for extra_cameras_idx in range(len(R_list)):
    sfm.drawRefSystem(ax, np.linalg.inv(sfm.ensamble_T(R_list[extra_cameras_idx], t_list[extra_cameras_idx])), '-', f'C{extra_cameras_idx+2}')
ax.scatter(current_3DPoints_opt.T[0, :], current_3DPoints_opt.T[1, :], current_3DPoints_opt.T[2, :], marker='o', color='g', s=5, label='3D Points')
xFakeBoundingBox = np.linspace(-6, 10, 2)
yFakeBoundingBox = np.linspace(-6, 10, 2)
zFakeBoundingBox = np.linspace(-6, 10, 2)
plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
plt.show()
plt.close(fig)


fig, ax = plt.subplots()
# sfm.drawRefSystem(ax, np.eye(4), '-', 'C1_ref', projection='2d')
# sfm.drawRefSystem(ax, np.linalg.inv(sfm.ensamble_T(R_old_opt, t_old_opt)), '-', 'C_OLD', projection='2d')
# for extra_cameras_idx in range(len(R_list)):
#     sfm.drawRefSystem(ax, np.linalg.inv(sfm.ensamble_T(R_list[extra_cameras_idx], t_list[extra_cameras_idx])), '-', f'C{extra_cameras_idx+2}', projection='2d')
ax.scatter(current_3DPoints_opt.T[0, :], current_3DPoints_opt.T[1, :], marker='o', color='g', s=5, label='3D Points')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.legend()
plt.show()
plt.close(fig)

