import numpy as np
import sfm
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
import os
import time



# Define the input file paths
input_dir = os.path.join(os.path.dirname(__file__), 'results')
R_file_path = os.path.join(input_dir, 'R_matrices.npy')
T_file_path = os.path.join(input_dir, 'T_vectors.npy')
points_3D_file_path = os.path.join(input_dir, '3D_points.npy')

# Load the R matrices, T vectors, and 3D points
R_file_path = os.path.join(input_dir, 'rotations.npz')
t_file_path = os.path.join(input_dir, 'translations.npz')
R_list, t_list = sfm.load_transformations(R_file_path, t_file_path)
current_3DPoints_opt = np.load(points_3D_file_path)

# ===================================
# DLT FOR POSE ESTIMATION OLD CAMERA
# ===================================

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