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


#####################
#  Plot 3D COLMAP   #
#####################
plt.figure(1)
ax = plt.axes(projection='3d', adjustable='box')
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')
ax.scatter((T_w1@colmap_3dpoints)[0, :], (T_w1@colmap_3dpoints)[1, :], (T_w1@colmap_3dpoints)[2, :], color='g', marker='.', label='Reconstructed 3D points')
i = 0
for cameras in T_gt.keys():
    if cameras == 'T_IMG_0001.png_diffReference':
        sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt[cameras]), '-', f'C_Old')
        # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.Old')
    else:       
        sfm.drawRefSystem(ax, T_w1@np.linalg.inv(T_gt[cameras]), '-', f'C_{i}')
        # fcv.drawRefSystem(ax, T_w1@ T_gt[cameras], '-.', f'C_.{i}')
    i+=1
#Matplotlib does not correctly manage the axis('equal')
xFakeBoundingBox = np.linspace(-10, 20, 2)
yFakeBoundingBox = np.linspace(-10, 20, 2)
zFakeBoundingBox = np.linspace(-10, 20, 2)
plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
plt.title('COLMAP 3D points')
plt.show()