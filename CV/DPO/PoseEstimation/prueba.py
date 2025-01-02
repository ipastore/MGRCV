import numpy as np
import sfm
import matplotlib.pyplot as plt
import cv2
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
import os


def get_image(image_id):
    image_path = os.path.join(os.path.dirname(__file__), f'../Images/Set_12MP/EntireSet/{image_id}.jpg')
    return plt.imread(image_path) 

def get_matrix(image_id):
    image_path = os.path.join(os.path.dirname(__file__), f'../Images/Set_12MP/EntireSet/{image_id}.jpg')
    return plt.imread(image_path) 

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
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No se encontró el archivo: {npz_path}")
    return np.load(npz_path)

data = load_npz_data('../RANSAC/results/inliers', REFERENCE_IMAGE, FIRST_IMAGE)

# Keypoints and matches for initial pair
ref_keypoints = data['keypoints0']
cam2_keypoints = data['keypoints1']
matches_pair2 = data['matches']
mask2= data['inliers_matches']

# Extract matched inliers
ref_2DPoints_pair2 = ref_keypoints[mask2[:, 0]]
cam2_2DPoints_pair2 = cam2_keypoints[mask2[:, 1]]
ref_idx_pair2 = mask2[:, 0]

# Convert to homogeneous coordinates
ref_3DPoints_h_pair2 = np.vstack((ref_2DPoints_pair2.T, np.ones(ref_2DPoints_pair2.T.shape[1])))
cam2_3DPoints_h_pair2 = np.vstack((cam2_2DPoints_pair2.T, np.ones(cam2_2DPoints_pair2.T.shape[1])))

# Image paths for visualization
ref_img = get_image(REFERENCE_IMAGE)
cam2_img = get_image(FIRST_IMAGE)

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
R_correct, t_correct, points_3D_initial = sfm.select_correct_pose(
    ref_3DPoints_h_pair2, cam2_3DPoints_h_pair2, K_camera, K_camera, R1, R2, t_candidates
)