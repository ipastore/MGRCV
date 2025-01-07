import sfm
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation as R

# Ruta al archivo points3D.txt
script_dir = os.path.dirname(__file__)
points3D_path = os.path.join(script_dir, "../colmap_project/output/points3D.txt")
camera_path = os.path.join(script_dir, "../colmap_project/output/images.txt")

# Cargar puntos 3D y poses de las cámaras
colmap_3dpoints = sfm.load_points_from_colmap(points3D_path)
colmap_camera_poses = sfm.extract_camera_poses(camera_path)
print(colmap_3dpoints.shape)
print(f"Poses de las cámaras cargadas: {len(colmap_camera_poses)}")
colmap_3dpoints = np.vstack([colmap_3dpoints.T, np.ones(colmap_3dpoints.shape[0])])

#########################
# Load Pose from COLMAP #
#########################
T_gt = {}
T_diffReference = {}

for camera_name in colmap_camera_poses.keys():
    R_gt = colmap_camera_poses[camera_name]['rotation_matrix']
    t_gt = colmap_camera_poses[camera_name]['translation_vector']
    T_gt[f'T_{camera_name}_gt'] = sfm.ensamble_T(R_gt, t_gt)

    R__, t__  = sfm.changePoseTransformation(R_gt, t_gt)
    T_diffReference[f'T_{camera_name}_diffReference'] = sfm.ensamble_T(R__, t__)

# Geometry
T_w1 = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]]).T


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


###########################
#  Align COLMAP with OWN  #
###########################
# Align
T_align = np.linalg.inv(T_gt['T_Img02_gt'])
T_0_gt = T_align @ T_gt['T_Img02_gt'] # == np.eye(4, 4)
T_1_gt = T_align @ T_gt['T_Img25_gt']
T_2_gt = T_align @ T_gt['T_Img15_gt']
T_3_gt = T_align @ T_gt['T_Img23_gt']

T_gt_ref = {}
for key in T_gt.keys():
    T_gt_ref[key] = T_gt[key] @ T_align
    

X_gt = np.linalg.inv(T_align) @ colmap_3dpoints


# Scale
T_1_own = sfm.ensamble_T(R_list[0], t_list[0])
T1_correcta = T_w1 @ T_1_own
scale_gt = np.linalg.norm(T_1_gt[:3, 3])
scale_own = np.linalg.norm(T_1_own[:3, 3])
scale_factor_1 =   scale_own / scale_gt
print(f'Scale factor with T_1: {scale_factor_1}')

T_2_own = sfm.ensamble_T(R_list[1], t_list[1])
T2_correcta = T_w1 @ T_2_own
T2_correcta_GT = T_w1 @ T_2_gt
scale_gt = np.linalg.norm(T_2_gt[:3, 3])
scale_own = np.linalg.norm(T_2_own[:3, 3])
scale_factor_2 =   scale_own / scale_gt
print(f'Scale factor with T_2: {scale_factor_2}')

T_3_own = sfm.ensamble_T(R_list[2], t_list[2])
scale_gt = np.linalg.norm(T_3_gt[:3, 3])
scale_own = np.linalg.norm(T_3_own[:3, 3])
scale_factor_3 =   scale_own / scale_gt
print(f'Scale factor with T_3: {scale_factor_3}')

# scale_factor = scale_factor_1
scale_factor = scale_factor_1
scale_factor_ref = scale_factor_1
X_gt[:3, :] *= scale_factor
# for key in T_gt_ref.keys():
#     T_gt_ref[key][:3, :] *= scale_factor
 
T_gt_scaled = {}

for key in T_gt_ref.keys():
    T_gt_scaled[key] = T_gt_ref[key].copy()
    T_gt_scaled[key][:3, :] *= scale_factor
    
T_0_gt[:3, 3] *= scale_factor
T_1_gt[:3, 3] *= scale_factor
T_2_gt[:3, 3] *= scale_factor
T_3_gt[:3, 3] *= scale_factor



#####################
#  Plot 3D COLMAP   #
#####################
current_3DPoints_opt = np.vstack([current_3DPoints_opt.T, np.ones(current_3DPoints_opt.shape[0])])


plt.figure(1)
ax = plt.axes(projection='3d', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter((T_w1@X_gt)[0, :], (T_w1@X_gt)[1, :], (T_w1@X_gt)[2, :], color='c', marker='.', label='Own 3D colmap')
ax.scatter((T_w1@current_3DPoints_opt)[0, :], (T_w1@current_3DPoints_opt)[1, :], (T_w1@current_3DPoints_opt)[2, :], color='g', marker='.', label='Mine')

# i = 0
# for cameras in T_gt_scaled.keys():
#     if cameras == 'T_IMG_0001.png_diffReference':
#         sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt_scaled[cameras]), '-', f'C_Old_GT')
#         # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.Old')
#     else:       
#         sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt_scaled[cameras]), '-', f'C_{i}_GT')
#         sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt_ref[cameras]), '-', f'C_{i}_NS')
#         # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.{i}')
#     i+=1
sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_0_gt), '-.', f'C_ref_gt')
sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_1_gt), '-.', f'C_2_gt')
sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_2_gt), '-.', f'C_3_gt')
sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_3_gt), '-.', f'C_4_gt')
sfm.drawRefSystem(ax, T_w1@np.eye(4), '-', 'C1_ref')
for extra_cameras_idx in range(len(R_list)):
    sfm.drawRefSystem(ax, T_w1@np.linalg.inv(sfm.ensamble_T(R_list[extra_cameras_idx], t_list[extra_cameras_idx])), '-', f'C{extra_cameras_idx+2}')
#Matplotlib does not correctly manage the axis('equal')
xFakeBoundingBox = np.linspace(-10, 20, 2)
yFakeBoundingBox = np.linspace(-10, 20, 2)
zFakeBoundingBox = np.linspace(-10, 20, 2)
plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
plt.title('COLMAP 3D points')
plt.show()

print('End')
# plt.figure(1)
# ax = plt.axes(projection='3d', adjustable='box')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')
# ax.set_zlabel('Z')
# ax.scatter((T_w1@colmap_3dpoints)[0, :], (T_w1@colmap_3dpoints)[1, :], (T_w1@colmap_3dpoints)[2, :], color='g', marker='.', label='Reconstructed 3D points')
# ax.scatter(current_3DPoints_opt.T[0, :], current_3DPoints_opt.T[1, :], current_3DPoints_opt.T[2, :], marker='o', color='r', label='3D Points')
# i = 0
# for cameras in T_gt.keys():
#     if cameras == 'T_IMG_0001.png_diffReference':
#         sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt[cameras]), '-', f'C_Old_GT')
#         # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.Old')
#     else:       
#         sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt[cameras]), '-', f'C_{i}_GT')
#         # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.{i}')
#     i+=1
# #Matplotlib does not correctly manage the axis('equal')
# xFakeBoundingBox = np.linspace(-10, 20, 2)
# yFakeBoundingBox = np.linspace(-10, 20, 2)
# zFakeBoundingBox = np.linspace(-10, 20, 2)
# plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
# plt.title('COLMAP 3D points')
# plt.show()

# ============================
# VISUALIZE 3D RECONSTRUCTION
# ============================
# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# sfm.drawRefSystem(ax, np.eye(4), '-', 'C1_ref')
# for extra_cameras_idx in range(len(R_list)):
#     sfm.drawRefSystem(ax, np.linalg.inv(sfm.ensamble_T(R_list[extra_cameras_idx], t_list[extra_cameras_idx])), '-', f'C{extra_cameras_idx+2}')
# ax.scatter(current_3DPoints_opt.T[0, :], current_3DPoints_opt.T[1, :], current_3DPoints_opt.T[2, :], marker='o', color='g', label='3D Points')
# xFakeBoundingBox = np.linspace(-6, 10, 2)
# yFakeBoundingBox = np.linspace(-6, 10, 2)
# zFakeBoundingBox = np.linspace(-6, 10, 2)
# plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
# plt.show()
# plt.close(fig)
