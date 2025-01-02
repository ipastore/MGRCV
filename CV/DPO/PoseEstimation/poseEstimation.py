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
sfm.visualize_residuals(ref_img, ref_2DPoints_pair2.T, ref_proj2DPoints_pair2, "Initial Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(cam2_img, cam2_2DPoints_pair2.T, cam2_proj2DPoints_pair2, "Initial Residuals in First Image", ax=axs[1])
plt.tight_layout()


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
t_c2_ref_opt = np.array([
    np.sin(t_theta_optimized) * np.cos(t_phi_optimized),
    np.sin(t_theta_optimized) * np.sin(t_phi_optimized),
    np.cos(t_theta_optimized)
])
global_3DPoints_opt = optimized_result.x[5:].reshape(3, -1).T
print("Updated number of ·D points:", global_3DPoints_opt.shape[0])

# ============================
# VISUALIZE OPTIMIZED RESULTS
# ============================

# Reconstruct optimized poses
R_c2_ref_opt = expm(sfm.crossMatrix(theta_c2_ref_opt))
T_c2_ref_opt = sfm.ensamble_T(R_c2_ref_opt, t_c2_ref_opt)
P_cam2_opt = sfm.get_projection_matrix(K_camera, T_c2_ref_opt)

# Project optimized points
global_3DPoints_h_opt = np.vstack((global_3DPoints_opt.T, np.ones((1, global_3DPoints_opt.shape[0]))))
ref_proj2DPoints_pair2_opt = sfm.project_to_camera(P_ref, global_3DPoints_h_opt)
cam2_proj2DPoints_pair2_opt = sfm.project_to_camera(P_cam2_opt, global_3DPoints_h_opt)

# Visualize optimized residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_2DPoints_pair2.T, ref_proj2DPoints_pair2_opt, "Optimized Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(cam2_img, cam2_2DPoints_pair2.T, cam2_proj2DPoints_pair2_opt, "Optimized Residuals in First Image", ax=axs[1])
plt.tight_layout()

# ============================
# PNP FOR ADDITIONAL CAMERAS
# ============================

def perform_pnp_and_triangulate(
    reference_image, target_image, K_C, point3D_map, optimized_3D_points, ref_cam_id, target_cam_id
):
    # Cargar correspondencias entre las cámaras
    pnp_data = np.load(os.path.join(os.path.dirname(__file__),f'../RANSAC/results/inliers/{reference_image}_vs_{target_image}_inliers.npz'))
    pnp_keypoints_ref = pnp_data['keypoints0']
    pnp_keypoints_new = pnp_data['keypoints1']
    pnp_mask = pnp_data['inliers_matches']

    # Filtrar puntos existentes y actualizar `point3D_map`
    existing_points = []
    new_matches = []

    for match in pnp_mask:
        ref_idx, cam3_idx = match[0], match[1]
        found = False
        for pid, cameras in point3D_map.items():
            if cameras.get(ref_cam_id) == ref_idx:  # Si el punto ya está en el diccionario
                point3D_map[pid][target_cam_id] = cam3_idx  # Actualizar keypoint en cam3
                existing_points.append(pid)
                found = True
                break
        if not found:
            new_matches.append(match)  # Matches que generan nuevos puntos 3D

    # Hacer PNP con puntos existentes
    if existing_points:
        # Obtener puntos 3D existentes y keypoints en cam3
        filtered_3D_points = np.array([optimized_3D_points[pid] for pid in existing_points])
        keypoints_cam3 = np.array([pnp_keypoints_new[point3D_map[pid][target_cam_id]] for pid in existing_points])
        
        # Convertir keypoints a coordenadas homogéneas
        x_cam3_h = np.vstack((keypoints_cam3.T, np.ones(keypoints_cam3.shape[0])))
        image_points = np.ascontiguousarray(x_cam3_h[0:2, :].T).reshape((x_cam3_h.shape[1], 1, 2))
        
        # Resolver PNP
        retval, rotation_vector, translation_vector = cv2.solvePnP(
            filtered_3D_points, image_points, K_C, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP
        )
        if not retval:
            raise ValueError("PNP falló al calcular la pose de la cámara")
        
        # Convertir a matriz de rotación
        R_pnp = expm(sfm.crossMatrix(rotation_vector))
        t_pnp = translation_vector.ravel()
    else:
        raise ValueError("No hay puntos existentes para calcular el PNP")

    # Triangular nuevos puntos 3D
    if new_matches:
        keypoints_ref_new = np.array([pnp_keypoints_ref[match[0]] for match in new_matches])
        keypoints_cam3_new = np.array([pnp_keypoints_new[match[1]] for match in new_matches])

        # Convertir keypoints a coordenadas homogéneas
        x_ref_h = np.vstack((keypoints_ref_new.T, np.ones(keypoints_ref_new.shape[0])))
        x_cam3_h = np.vstack((keypoints_cam3_new.T, np.ones(keypoints_cam3_new.shape[0])))

        # Matrices de proyección para ref y cam3
        P_ref = sfm.get_projection_matrix(K_C, np.eye(4))
        T_cam3_to_ref = sfm.ensamble_T(R_pnp, t_pnp)
        P_cam3 = sfm.get_projection_matrix(K_C, T_cam3_to_ref)

        # Triangular nuevos puntos 3D
        new_points_3D = sfm.triangulate_points(x_ref_h, x_cam3_h, P_ref, P_cam3)
        new_points_3D = new_points_3D[:3, :].T.reshape(-1, 3)

        # Actualizar el diccionario con los nuevos puntos
        for i, match in enumerate(new_matches):
            new_point_id = len(point3D_map)  # Generar nuevo ID para el punto 3D
            point3D_map[new_point_id] = {ref_cam_id: match[0], target_cam_id: match[1]}
            optimized_3D_points = np.vstack((optimized_3D_points, new_points_3D[i,:]))

    return R_pnp, t_pnp, point3D_map, optimized_3D_points


R_cam3, t_cam3, point3D_map, global_3DPoints_updated = perform_pnp_and_triangulate(
    REFERENCE_IMAGE, AVAILABLE_IMAGES[0], K_camera, point3D_map, global_3DPoints_opt, ref_cam_id=1, target_cam_id=3
)

print("Rotation matrix (Camera 3 with respect to Camera 1):\n", R_cam3)
print("Translation vector (Camera 3 with respect to Camera 1):\n", t_cam3)
print("Number of 3D points after PnP and triangulation:", global_3DPoints_updated.shape[0])


# ================================
# VISUALIZATION OF INITIAL RESULTS
# ================================

cam3_img = get_image(AVAILABLE_IMAGES[0])

# Projection matrices
T_cam3_to_ref = sfm.ensamble_T(R_cam3, t_cam3)
P_cam3_NotOpt = sfm.get_projection_matrix(K_camera, T_cam3_to_ref)

# Project points into cameras
# Filtrar los puntos visibles en ambas cámaras
visible_points = [
    pid for pid, cameras in point3D_map.items() if 1 in cameras and 3 in cameras
]
# Proyección en coordenadas homogéneas para los puntos visibles
pair3_3DPoints_h = np.vstack((
    np.array([global_3DPoints_updated[pid] for pid in visible_points]).T,
    np.ones(len(visible_points))
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

ref_proj2DPoints_pair3 = sfm.project_to_camera(P_ref, pair3_3DPoints_h)
cam3_proj2DPoints_pair3 = sfm.project_to_camera(P_cam3_NotOpt, pair3_3DPoints_h)
# Obtener los keypoints correspondientes en ambas cámaras
ref_FilteredKeypoints_pair3 = np.array([ref_keypoints[point3D_map[pid][1]] for pid in visible_points])
cam3_FilteredKeypoints_pair3 = np.array([cam3_keypoints[point3D_map[pid][3]] for pid in visible_points])


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

# def prepare_op_vector(cameras, R_list, t_list, points_3D):
#     """
#     Prepara el vector inicial de parámetros para el Bundle Adjustment para n cámaras.

#     Parameters:
#         cameras (list): Lista de IDs de las cámaras.
#         R_list (list): Lista de matrices de rotación de cada cámara.
#         t_list (list): Lista de vectores de traslación de cada cámara.
#         points_3D (np.array): Coordenadas 3D de los puntos visibles.

#     Returns:
#         np.array: Vector inicial de parámetros.
#     """
#     op_vector = []

#     # Añadir parámetros de las cámaras (rotación y traslación)
#     for cam_id, (R, t) in enumerate(zip(R_list, t_list)):
#         theta = sfm.crossMatrixInv(logm(R.astype('float64')))
#         op_vector.extend([*theta, *t])

#     # Añadir coordenadas de los puntos 3D
#     op_vector.extend(points_3D.flatten())
#     return np.array(op_vector)


def prepare_op_vector(R_cameras, t_cameras, points_3D_initial):
    """
    Prepara el vector de parámetros para la optimización (op_vector).

    Args:
        R_cameras (list of np.ndarray): Lista de matrices de rotación (3x3) para todas las cámaras (excluyendo la de referencia).
        t_cameras (list of np.ndarray): Lista de vectores de traslación (3,) para todas las cámaras (excluyendo la de referencia).
        points_3D_initial (np.ndarray): Matriz (3xN) con las coordenadas iniciales de los puntos 3D.

    Returns:
        np.ndarray: Vector concatenado de parámetros iniciales para la optimización.
    """
    op_vector = []

    # Rotación y traslación de la cámara 2 (usando coordenadas polares para la traslación)
    R_cam2 = R_cameras[0]  # Primera cámara adicional respecto a la referencia
    t_cam2 = t_cameras[0]

    initial_guess_theta_cam2 = sfm.crossMatrixInv(logm(R_cam2.astype('float64')))
    t_norm_cam2 = np.linalg.norm(t_cam2)
    t_theta_cam2 = np.arccos(t_cam2[2] / t_norm_cam2)
    t_phi_cam2 = np.arctan2(t_cam2[1], t_cam2[0])

    op_vector.extend(initial_guess_theta_cam2)
    op_vector.append(t_theta_cam2)
    op_vector.append(t_phi_cam2)

    # Rotaciones y traslaciones para cámaras adicionales (en coordenadas cartesianas)
    for i in range(1, len(R_cameras)):
        R_cam = R_cameras[i]
        t_cam = t_cameras[i]

        # Representación logarítmica para la rotación
        initial_guess_theta_cam = sfm.crossMatrixInv(logm(R_cam.astype('float64')))
        op_vector.extend(initial_guess_theta_cam)

        # Traslación en coordenadas cartesianas
        op_vector.extend(t_cam)

    # Puntos 3D en coordenadas cartesianas
    op_vector.extend(points_3D_initial.flatten())

    return np.array(op_vector)


def recover_parameters(op_vector, n_cameras, n_points):
    """
    Recupera las rotaciones, traslaciones y puntos 3D del vector optimizado.

    Args:
        op_vector (np.ndarray): Vector de parámetros optimizados.
        n_cameras (int): Número total de cámaras (incluyendo la de referencia).
        n_points (int): Número total de puntos 3D.

    Returns:
        dict: Diccionario con rotaciones, traslaciones y puntos 3D:
            {
                "rotations": list of np.ndarray,  # Lista de vectores de rotación (logm)
                "translations": list of np.ndarray,  # Lista de vectores de traslación (3,)
                "points_3D": np.ndarray  # Matriz de puntos 3D optimizados (n_points x 3)
            }
    """
    offset = 0
    rotations = []
    translations = []

    # **Cámara de referencia** (fija, no está en el vector optimizado)
    rotations.append(np.zeros(3))  # Rotación identidad (logm(R_ref) = [0, 0, 0])
    translations.append(np.zeros(3))  # Traslación fija en el origen [0, 0, 0]

    # **Cámara 2 (en coordenadas polares)**
    theta_c2_ref = op_vector[offset:offset+3]
    t_theta = op_vector[offset+3]
    t_phi = op_vector[offset+4]
    t_c2_ref = np.array([
        np.sin(t_theta) * np.cos(t_phi),
        np.sin(t_theta) * np.sin(t_phi),
        np.cos(t_theta)
    ])
    rotations.append(theta_c2_ref)
    translations.append(t_c2_ref)
    offset += 5

    # **Cámaras adicionales (en cartesiano)**
    for _ in range(3, n_cameras + 1):  # Desde cam3 en adelante
        theta = op_vector[offset:offset+3]
        t = op_vector[offset+3:offset+6]
        rotations.append(theta)
        translations.append(t)
        offset += 6
    
    # **Puntos 3D**
    points_3D = op_vector[offset:].reshape(3, n_points).T

    return {
        "rotations": rotations,        # Rotaciones optimizadas (logm)
        "translations": translations,  # Traslaciones optimizadas
        "points_3D": points_3D         # Coordenadas optimizadas de puntos 3D
    }

def resBundleProjection_multicamera(Op, point3D_map, keypoints_2D, K_c, n_cameras):
    """
    Calcula los residuales por cámara para el Bundle Adjustment.

    Args:
        Op (np.ndarray): Vector de parámetros optimizados.
        point3D_map (dict): Diccionario que mapea puntos 3D a cámaras y keypoints.
        keypoints_2D (list of np.ndarray): Keypoints observados para cada cámara.
        K_c (np.ndarray): Matriz intrínseca de la cámara.
        n_cameras (int): Número total de cámaras (incluyendo la de referencia).

    Returns:
        np.ndarray: Vector de residuales concatenados.
    """
    # Extraer rotaciones, traslaciones y puntos 3D del vector Op
    offset = 0
    rotations = []
    translations = []

    # Cámara 2 (rotación y traslación en polares)
    theta_cam2 = Op[offset:offset+3]
    t_theta_cam2 = Op[offset+3]
    t_phi_cam2 = Op[offset+4]
    t_cam2 = np.array([
        np.sin(t_theta_cam2) * np.cos(t_phi_cam2),
        np.sin(t_theta_cam2) * np.sin(t_phi_cam2),
        np.cos(t_theta_cam2)
    ])
    rotations.append(theta_cam2)
    translations.append(t_cam2)
    offset += 5

    # Cámaras adicionales (cartesianas)
    for i in range(2, n_cameras):
        theta = Op[offset:offset+3]
        t = Op[offset+3:offset+6]
        rotations.append(theta)
        translations.append(t)
        offset += 6

    # Puntos 3D
    points_3D = Op[offset:].reshape(3, -1)

    # Inicializar vector de residuales
    residuals = []

    # **Calcular residuales para cámaras adicionales**
    for cam_idx in range(1, n_cameras + 1):  # Cam2 es 2, Cam3 es 3, etc.
        
        if cam_idx == 1:
            P = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Matriz de proyección fija
        else:
            R = expm(sfm.crossMatrix(rotations[cam_idx - 2]))
            t = translations[cam_idx - 2].reshape(-1, 1)

            # Matriz de proyección
            T = np.hstack((R, t))
            P = K_c @ T

        # **1. Filtrado de puntos visibles**
        visible_points = [
            (point_idx, cam_info[cam_idx])
            for point_idx, cam_info in point3D_map.items()
            if cam_idx in cam_info
        ]

        if not visible_points:
            continue  # Si no hay puntos visibles, saltar esta cámara

        # Crear arrays de puntos y keypoints
        point_indices = [point_idx for point_idx, _ in visible_points]
        keypoint_indices = [keypoint_idx for _, keypoint_idx in visible_points]

        points_3D_visible = points_3D[:, point_indices]  # (3, N)
        keypoints_2D_visible = keypoints_2D[cam_idx - 1][:, keypoint_indices]  # (2, N)

        # **2. Calcular residuales**
        points_h = np.vstack((points_3D_visible, np.ones((1, points_3D_visible.shape[1]))))  # (4, N)
        projected_2D = sfm.project_to_camera(P, points_h)  # (2, N)

        # Residuales (vectorizados)
        residuals_2D = keypoints_2D_visible - projected_2D[0:2,:]
        residuals.extend(residuals_2D.flatten())

    return np.array(residuals)


R_list = [R_cam2, R_cam3]
t_list = [t_cam2, t_cam3]

# Transformar al formato esperado
keypoints_2D = [
    ref_keypoints.T,
    cam2_keypoints.T,
    cam3_keypoints.T   
]

# Preparar el vector de parámetros
op_vector = prepare_op_vector(R_list, t_list, global_3DPoints_updated.T)
print(f"Vector de parámetros para optimización: {op_vector.shape}")


# Ejecutar la optimización
optimized_result = least_squares(
    resBundleProjection_multicamera,
    op_vector,
    args=(point3D_map, keypoints_2D, K_camera, len(keypoints_2D)),
    method='lm'  # Levenberg-Marquardt (puedes ajustar según tu problema)
)

# Datos de entrada
n_cameras = 3  # Por ejemplo, 3 cámaras (ref, cam2, cam3)
n_points = global_3DPoints_updated.shape[0]  # Por ejemplo, 5073 puntos 3D

# Recuperar parámetros optimizados
optimized_data = recover_parameters(optimized_result.x, n_cameras, n_points)

# Acceder a los datos recuperados
rotations = optimized_data["rotations"]  # Lista de rotaciones (logm)
translations = optimized_data["translations"]  # Lista de traslaciones (3,)
global_3DPoints_opt = optimized_data["points_3D"]  # Puntos 3D optimizados (n_points x 3)

# Ejemplo: Imprimir resultados
print("Rotaciones optimizadas (logm):", rotations)
print("Traslaciones optimizadas:", translations)
print("Puntos 3D optimizados (shape):", global_3DPoints_opt.shape)
print("Magnitude of the translation vector:", np.linalg.norm(translations[1]))

# ============================
# VISUALIZE OPTIMIZED RESULTS
# ============================

# Reconstruct optimized poses
R_cam3 = expm(sfm.crossMatrix(rotations[2]))
T_cam3_to_ref = sfm.ensamble_T(R_cam3, translations[2])
P_cam3_opt = sfm.get_projection_matrix(K_camera, T_cam3_to_ref)

# Cargar correspondencias entre las cámaras
data_pair3 = np.load(os.path.join(os.path.dirname(__file__),f'../RANSAC/results/inliers/{REFERENCE_IMAGE}_vs_{AVAILABLE_IMAGES[0]}_inliers.npz'))
cam3_keypoints = data_pair3['keypoints1']
del data_pair3

# Project optimized points
new_global_3DPoints_h_opt = np.vstack((global_3DPoints_opt, np.ones((1, global_3DPoints_opt.shape[1]))))
ref_proj2DPoints_pair3_opt = sfm.project_to_camera(P_ref, global_3DPoints_h_opt)
cam3_proj2DPoints_pair3_opt = sfm.project_to_camera(P_cam3_opt, global_3DPoints_h_opt)

# Visualize optimized residuals
fig, axs = plt.subplots(1, 2, figsize=(18, 6))
sfm.visualize_residuals(ref_img, ref_2DPoints_pair3.T, ref_proj2DPoints_pair3_opt, "Optimized Residuals in Reference Image", ax=axs[0])
sfm.visualize_residuals(cam3_img, cam3_2DPoints_pair3.T, cam3_proj2DPoints_pair3_opt, "Optimized Residuals in First Image", ax=axs[1])
plt.tight_layout()
plt.show()
plt.close(fig)

print("FINISIHED")