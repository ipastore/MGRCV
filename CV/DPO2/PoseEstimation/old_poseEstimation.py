import numpy as np
import sfm
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import expm, logm
from scipy.optimize import least_squares

#### POSE ESTIMATION FROM TWO VIEWS ###

reference_image_name = 'Img02_Try1_12M'
first_image_name = 'Img25_Try1_12M'
available_images = ['Img14_Try1_12M']
                    # 'Img02_Try1_12M',
                    # 'Img14_Try1_12M',
                    # 'Img23_Try1_12M']

npz_file = f'../RANSAC/results/inliers/{reference_image_name}_vs_{first_image_name}_inliers.npz'
data = np.load(npz_file)
keypoints1 = data['keypoints0']
keypoints2 = data['keypoints1']
matches = data['matches']
inliers_mask = data['inliers_matches']
matched_inliers1 = keypoints1[matches[:, 0]] 
matched_inliers2 = keypoints2[matches[:, 1]] 
x1_h = np.vstack((matched_inliers1.T, np.ones(matched_inliers1.T.shape[1])))
x2_h = np.vstack((matched_inliers2.T, np.ones(matched_inliers2.T.shape[1])))


ref_img_path = f'../Images/Set_12MP/EntireSet/{reference_image_name}.jpg'
first_img_path = f'../Images/Set_12MP/EntireSet/{first_image_name}.jpg'

ref_img_file = plt.imread(ref_img_path)
first_img_file = plt.imread(first_img_path)


F_file = f'../RANSAC/results/fundamental/F_{reference_image_name}_vs_{first_image_name}.txt'
F_est = sfm.load_matrix(F_file)
sfm.visualize_epipolar_lines(F_est, ref_img_file, first_img_file, show_epipoles=True, automatic=False)


K_file = '../Camera_calibration/Calibration_12MP/K_Calibration_12MP.txt'
K_cam = sfm.load_matrix(K_file)
K1 = K2 = K_C = K_cam
E = sfm.compute_essential_matrix_from_F(F_est, K1, K2)
print("E estimation:")
print(E)

R1, R2, t = sfm.decompose_essential_matrix(E)
R_c2_c1_initial, t_c2_c1_initial, X_c1_w_initial = sfm.select_correct_pose(x1_h, x2_h, K_C, K_C, R1, R2, t)

print("R_correct (Relative):")
print(R_c2_c1_initial)
print("t_correct (Relative):")
print(t_c2_c1_initial)


######################################################
################## BUNDLE AJUSTMENT ##################
######################################################
# View residuals from initial calculations
#From World to image1
P_c1_c1_initial = sfm.get_projection_matrix(K_C, np.eye(4))
x1_proj_initial = sfm.project_to_camera(P_c1_c1_initial, X_c1_w_initial)

#From World to image2
T_c2_c1_initial = sfm.ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
P_c2_c1_initial = sfm.get_projection_matrix(K_C,T_c2_c1_initial)
x2_proj_initial = sfm.project_to_camera(P_c2_c1_initial, X_c1_w_initial)

fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img_file, matched_inliers1.T, x1_proj_initial, "Initial Residuals in Image 1", ax=axs[0])
sfm.visualize_residuals(first_img_file, matched_inliers2.T, x2_proj_initial, 'Initial Residuals in Image 2', ax=axs[1])
plt.tight_layout()
plt.show()

## Optimization
# Initialize the initial guess with correct values
x1_data = matched_inliers1.T
x2_data = matched_inliers2.T
theta_c2_c1_initial = sfm.crossMatrixInv(logm(R_c2_c1_initial.astype('float64')))

t_norm = np.linalg.norm(t_c2_c1_initial, axis=-1)
t_theta = np.arccos(t[2]/t_norm)
t_phi = np.arctan2(t[1], t[0])

intial_guess = np.hstack((theta_c2_c1_initial, t_theta, t_phi, X_c1_w_initial[:3, :].flatten())) 
nPoints = X_c1_w_initial.shape[1]
optimized = least_squares(sfm.resBundleProjection, 
                intial_guess, 
                args=(x1_data, x2_data, K_C, nPoints),
                method='lm'
                )

# Extract optimized parameters
theta_c2_c1_opt = optimized.x[:3]
# t_c2_c1_opt = optimized.x[3:5]
t_theta = optimized.x[3]
t_phi = optimized.x[4]
t_c2_c1_opt = np.array([np.sin(t_theta)*np.cos(t_phi), np.sin(t_theta)*np.sin(t_phi), np.cos(t_theta)])
X_c1_w_opt = optimized.x[5:].reshape(3, -1).T

## Visualize optimization
R_c2_c1_opt = expm(sfm.crossMatrix(theta_c2_c1_opt))
T_c2_c1_opt = sfm.ensamble_T(R_c2_c1_opt, t_c2_c1_opt)
P_c2_c1_opt = sfm.get_projection_matrix(K_C, T_c2_c1_opt)

X_c1_w_opt_h = np.vstack((X_c1_w_opt.T, np.ones((1, X_c1_w_opt.shape[0]))))

x1_proj_opt = sfm.project_to_camera(P_c1_c1_initial, X_c1_w_opt_h)
x2_proj_opt = sfm.project_to_camera(P_c2_c1_opt, X_c1_w_opt_h)

# View residuals from optimized calculations
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img_file, x1_data, x1_proj_opt, "Optimized Residuals in Image 1", ax=axs[0])
sfm.visualize_residuals(first_img_file, x2_data, x2_proj_opt, 'Optimized Residuals in Image 2', ax=axs[1])
plt.tight_layout()
plt.show()


######################################################
################## PNP ALGORITHM #####################
######################################################

# # First I must charge the image and the 2D points
# pnp_img_path = f'../Images/Set_12MP/EntireSet/{available_images[0]}.jpg'
# pnp_img_file = plt.imread(available_images[0])

# # Points
# npz_file = f'../RANSAC/results/inliers/{reference_image_name}_vs_{available_images[0]}_inliers.npz'
# data = np.load(npz_file)
# new_keypoints1 = data['keypoints0']
# new_keypoints2 = data['keypoints1']
# new_matches = data['matches']
# new_inliers_mask = data['inliers_matches']
# new_matched_inliers1 = keypoints1[matches[:, 0]] 
# new_matched_inliers2 = keypoints2[matches[:, 1]] 

# # Now we must filter just the poins already triangulated before
# previous_inliers_indices = set(inliers_mask[:, 0])
# filtered_new_inliers_mask = np.array([match for match in new_inliers_mask if match[0] in previous_inliers_indices])

# imagePoints = np.ascontiguousarray(x1_h[0:2, :].T).reshape((x1_h.shape[1], 1, 2))
# retval, theta_c3_c1_initial, t_c3_c1_initial = cv2.solvePnP(X_c1_w_opt, imagePoints, K_C, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)
