import numpy as np
import os
import matplotlib.pyplot as plt
import scipy.optimize as scOptim
import cv2

from utils.drawingCV import *
from utils.matrixOperationsCV import *
from utils.bundleAdjustmentCV import *


def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)
        
        
def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {str(e)}")

def ensure_column_vector(array):
    return array.reshape(-1, 1) if array.ndim == 1 else array

def check_array_dimensions(array, rows, cols):
    if array.shape != (rows, cols):
        raise ValueError(f"Array shape mismatch: Expected {rows}x{cols}, got {array.shape}.")

def compute_radial_distance(points):
    return np.sqrt(points[0]**2 + points[1]**2)

def compute_distorsion_radiale(theta, D_k_array):
    d_theta = theta + theta**3 * D_k_array[0] + theta**5 * D_k_array[1] + theta**7 * D_k_array[2] + theta**9 * D_k_array[3]
    return d_theta

def compute_Kannala_Brandt_projection(Kc, d_theta, phi):
    """
    Proyecta puntos utilizando el modelo Kannala-Brandt.
    
    Parámetros:
    - Kc: Matriz de calibración intrínseca (3x3).
    - d_theta: Ángulo corregido con distorsión radial (array escalar o 1D).
    - phi: Ángulo azimutal (array escalar o 1D).

    Retorno:
    - u: Coordenadas proyectadas (3xN o 3x1).
    """
    
    if np.isscalar(d_theta): 
        A = np.array([[d_theta * np.cos(phi)],
                      [d_theta * np.sin(phi)],
                      [1]])
    else: 
        A = np.vstack((
            d_theta * np.cos(phi),
            d_theta * np.sin(phi),
            np.ones_like(d_theta) 
        ))
    
    return Kc @ A

def kannala_forward_model(x, K_c, D_k):
    """
    Proyecta puntos 3D en la imagen utilizando el modelo Kannala-Brandt.
    
    Parámetros:
    - x: Punto(s) 3D en coordenadas homogéneas (3xN o 3x1).
    - K_c: Matriz de calibración intrínseca de la cámara (3x3).
    - D_k: Coeficientes de distorsión radial.

    Retorno:
    - u: Coordenadas proyectadas en la imagen (3xN o 3x1).
    """    
    
    # Asegurar que la entrada es una matriz 3xN
    if x.ndim == 1 or x.shape[1] == 1:
        x = ensure_column_vector(x)  # Si es un solo vector, lo convierte en columna
    elif x.shape[0] != 4:
        raise ValueError(f"Los puntos deben estar en coordenadas 3xN o 3x1. Se recibió una matriz de tamaño {x.shape}.")
    
    # Step 1: Compute the radial distance
    r_dist = compute_radial_distance(x)
    
    # Step 2: Compute the theta angle
    theta = np.arctan2(r_dist[0], x[2])
    phi = np.arctan2(x[1], x[0])  # Ángulo azimutal

    # Step 3: Calcule la distorsion radiale
    d_theta = compute_distorsion_radiale(theta, D_k)

    # Step 4: Compute the projection
    u = compute_Kannala_Brandt_projection(K_c, d_theta, phi)
    
    # # Paso 1: Calcular la distancia radial para cada punto
    # r_dist = np.sqrt(x[0, :]**2 + x[1, :]**2)  # Vector de distancias radiales

    # # Paso 2: Calcular el ángulo theta
    # theta = np.arctan2(r_dist, x[2, :])  # Ángulo de elevación

    # # Paso 3: Calcular el ángulo phi
    # phi = np.arctan2(x[1, :], x[0, :])  # Ángulo azimutal

    # # Paso 4: Calcular la distorsión radial
    # d_theta = compute_distorsion_radiale(theta, D_k)  # Distorsión radial

    # # Paso 5: Proyectar los puntos utilizando la matriz de calibración
    # u = compute_Kannala_Brandt_projection(K_c, d_theta, phi)  # Coordenadas proyectadas

    return u

def kannala_backward_model(u, K_c, D_k):
    """
    Implements the Kannala-Brandt unprojection model.
    Inputs:
        u: 2D homogeneous coordinates on the image (3x1 column vector).
        K_c: Camera calibration matrix (3x3).
        D: Distortion coefficients [k1, k2, k3, k4].
    Outputs:
        v: Ray in the camera frame (unit vector, 3x1 column vector).
    """
    u = ensure_column_vector(u)
    
    # Step 2: Transform to normalized image coordinates
    x_1 = np.linalg.inv(K_c) @ u  # Result is a (3, 1) column vector

    # Step 3: Compute radial distance d and azimuthal angle phi
    d = np.sqrt(x_1[0, 0]**2 + x_1[1, 0]**2) / x_1[2, 0]
    phi = np.arctan2(x_1[1, 0], x_1[0, 0])

    # Step 4: Solve for theta using the distortion polynomial
    theta_solutions = np.roots([D_k[3], 0, D_k[2], 0, D_k[1], 0, D_k[0], 0, 1, -d])
    
    # Step 5: Filter valid theta (real, positive, within range)
    theta = None
    for theta_value in theta_solutions:
        if np.isreal(theta_value) and theta_value.real > 0:
            theta = theta_value.real
            break

    if theta is None:
        raise ValueError("No valid theta found for unprojection.")
    
    
    # Step 6: Compute 3D ray in camera coordinates
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Construct the ray as a (3, 1) column vector
    # v = np.array([
    #     [sin_theta * cos_phi],
    #     [sin_theta * sin_phi],
    #     [cos_theta]
    # ])
    
    v = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ])
    
    return v


def calculate_triangulation_planes(v):
    
    # Check the dimensions of the input 
    if v.shape != (3,):
        raise ValueError(f"Input vector v must be of shape (3,), but got shape {v.shape}")
    
    # Compute the plane symmetric containing the ray and Z axis
    plane_sym = np.array([-v[1], v[0], 0, 0])
    
    # Compute the plane perpendicular containing the ray and being perpendicular to the sym plane
    plane_perp = np.array([-v[2] * v[0], -v[2] * v[1], v[0]**2 + v[1]**2, 0])

    return plane_sym, plane_perp
    

def triangulation_kannala(x1, x2, K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2):
    """
    Realiza la triangulación de un punto a partir de dos imágenes utilizando el modelo Kannala-Brandt.
    Parámetros:
    - x1: Coordenadas homogéneas 2D en la imagen de la cámara 1 (3x1 columna).
    - x2: Coordenadas homogéneas 2D en la imagen de la cámara 2 (3x1 columna).
    - K_1: Matriz de calibración de la cámara 1 (3x3).
    - K_2: Matriz de calibración de la cámara 2 (3x3).
    - D1_k_array: Coeficientes de distorsión de la cámara 1 [k1, k2, k3, k4].
    - D2_k_array: Coeficientes de distorsión de la cámara 2 [k1, k2, k3, k4].
    - T_w_c1: Transformación de la cámara 1 al marco del mundo (4x4).
    - T_w_c2: Transformación de la cámara 2 al marco del mundo (4x4).
    Retorna:
    - X_tri_FRW: Coordenadas trianguladas en el marco del mundo (4x1).
    """

    assert x1.size == x2.size, "x1 and x2 must have the same number of points for Kannala triangulation."
    n_points = x1.shape[1]
    X_tri_FRW = np.zeros((4, x1.shape[1]))

    for i in range(n_points):
        x_1 = x1[:,i]
        x_2 = x2[:,i]
        v1 = kannala_backward_model(x_1, K_1, D1_k_array)
        v2 = kannala_backward_model(x_2, K_2, D2_k_array)
        debug_print("\nX1: ", x_1)
        debug_print("X2: ", x_2)
        debug_print("V1: ", v1)
        debug_print("V2: ", v2)

        plane_sym_1, plane_perp_1 = calculate_triangulation_planes(v1)
        plane_sym_2, plane_perp_2 = calculate_triangulation_planes(v2)
        debug_print("\nPlane sym 1 - FR1: ", plane_sym_1)
        debug_print("Plane perp 1 - FR1: ", plane_perp_1)
        debug_print("Plane sym 2 - FR2: ", plane_sym_2)
        debug_print("Plane perp 2 - FR2: ", plane_perp_2, "\n")
        
        # Transformar los planos de la cámara 1 al marco de la cámara 2
        T_c1_c2 = np.linalg.inv(T_w_c1) @ T_w_c2
        debug_print("T_c1_c2: ", T_c1_c2, sep="\n")
        plane_sym_1_fr2 = T_c1_c2.T @ plane_sym_1
        plane_perp_1_fr2 = T_c1_c2.T @ plane_perp_1
        
        debug_print("\nPlane sym 1 - FR2: ", plane_sym_1_fr2)
        debug_print("Plane perp 1 - FR2: ", plane_perp_1_fr2, "\n")
        
        A = np.vstack([
            plane_sym_1_fr2,
            plane_perp_1_fr2,
            plane_sym_2,
            plane_perp_2
        ])
        
        debug_print("A - FR2: ", A, sep="\n")
        # Paso 4: Resolver AX = 0 usando SVD
        _, _, Vt = np.linalg.svd(A)
        debug_print("Vt: ", Vt, sep="\n")
        X_tri_FR2 = Vt[-1]  # Último vector singular
        X_tri_FR2 /= X_tri_FR2[3]
        debug_print("\nX_tri_FR2: ", X_tri_FR2)
        X_tri_FRW[:,i] = T_w_c2 @ X_tri_FR2
        debug_print(f"Point {i}: 3D Coord: {X_tri_FRW[:, i].flatten()}")

    
    return X_tri_FRW

def resBundleProjection(Op, x_data, T_w_c1, T_w_c2, K_1, K_2, D1_k_array, D2_k_array, nPairs=2):
    """
    Compute residuals between observed 2D points and projected 3D points.
    
    Parameters:
        Op: Optimization parameters (rotation, translation, 3D points).
        x_data: Observed 2D points (array of shape 2x[nPairs * nPoints]).
        T_wc1: Pose of camera 1 in world coordinates (4x4).
        T_wc2: Pose of camera 2 in world coordinates (4x4).
        K_1: Camera 1 intrinsic calibration matrix (3x3).
        K_2: Camera 2 intrinsic calibration matrix (3x3).
        D1_k_array: Distortion coefficients for camera 1.
        D2_k_array: Distortion coefficients for camera 2.
        nPairs: Number of camera pairs.
        
    Returns:
        res: Residuals (difference between observed and reprojected points).
    """

    # --- Extract Parameters ---
    posStartX = 6 * (nPairs - 1)  # 6 parameters (rotation + translation) per pair (except the first)
    nPoints = int((Op.shape[0] - posStartX) / 3)  # Number of 3D points
    X_3D = np.vstack([Op[posStartX:].reshape(3, nPoints), np.ones((1, nPoints))])  # Convert to homogeneous coordinates

    # --- Initial Projections for Camera Pair 1 ---
    T_c1_w = np.linalg.inv(T_w_c1)  # Camera 1 relative to world
    T_c2_w = np.linalg.inv(T_w_c2)  # Camera 2 relative to world
    u_1_array = np.empty([3, nPoints])
    u_2_array = np.empty([3, nPoints])
    for i in range(nPoints):
        x_3d = X_3D[:,i]                       
        x_3D_C1 = T_c1_w @ x_3d.T
        x_3D_C2 = T_c2_w @ x_3d.T
        u_1 = kannala_forward_model(x_3D_C1, K_1, D1_k_array).flatten()      
        u_2 = kannala_forward_model(x_3D_C2, K_2, D2_k_array).flatten()
        u_1_array[:, i] = u_1
        u_2_array[:, i] = u_2

    res = []
    for j in range(nPoints):        
        res.append(x_data[0, j] - u_1_array[0, j])        
        res.append(x_data[1, j] - u_1_array[1, j])        
        res.append(x_data[0, j+nPoints] - u_2_array[0, j])
        res.append(x_data[1, j+nPoints] - u_2_array[1, j])
        
    for i in range(nPairs-1):
        theta_rot = Op[i*6:i*6+3]
        tras = Op[i*6+3:i*6+6]
        T_wAwB = ObtainPose(theta_rot, tras[0], tras[1])    # traslation between pair i and original pair
        T_wAwB[0:3,3] = Op[i*6+3:i*6+6]        
        for j in range(nPoints):            
            x_3d = X_3D[:, j]
            x_3d_B = T_wAwB @ x_3d.T
            x_3d_1B = T_c1_w @ x_3d_B.T           
            x_3d_2B = T_c2_w @ x_3d_B.T  
            u_1 = kannala_forward_model(x_3d_1B, K_1, D1_k_array).flatten()
            u_2 = kannala_forward_model(x_3d_2B, K_2, D2_k_array).flatten()
                 
            res.append(x_data[0, j+2*i*nPoints+2*nPoints] - u_1[0])
            res.append(x_data[1, j+2*i*nPoints+2*nPoints] - u_1[1])            
            res.append(x_data[0, j+2*i*nPoints+2*nPoints+nPoints] - u_2[0])
            res.append(x_data[1, j+2*i*nPoints+2*nPoints+nPoints] - u_2[1])
    return res

# def resBundleProjection(Op, x_data, T_w_c1, T_w_c2, K_1, K_2, D1_k_array, D2_k_array, nPairs=2):
#     """
#     Input:
#         Op: parameters to optimize
#         x_data: 2d points
#         T_wc1: camera 1 pose in the world frame
#         T_wc2: camera 2 pose in the world frame
#         K_1: camera 1 calibration matrix
#         K_2: camera 2 calibration matrix
#         D1_k_array: camera 1 distortion coefficients
#         D2_k_array: camera 2 distortion coefficients
#         nPairs: number of pairs of cameras
#     Output:
#         res: residuals
#     """
#     posStartX = 6 * (nPairs - 1)    # you pass 5 params for each pair of cameras and dont pass the first pair
    
#     nPoints = int((Op.shape[0] - posStartX)/3)   # get the number of 3d points
#     x_3dp = Op[posStartX:].reshape((3, nPoints))    # get the 3d points
    
#     #x_3dp = x_3dp.T   # reshape the 3d points
#     x_3dp = np.vstack([x_3dp, np.ones((1, nPoints))])    # add the 1s to the 3d points
    
#     # Project the triangulated point to the camera 1
#     T_c1_w = np.linalg.inv(T_w_c1)
#     T_c2_w = np.linalg.inv(T_w_c2)
#     u_1_array, u_2_array = project_points_to_cameras_kannala(x_3dp, T_c1_w, T_c2_w, K_1, K_2, D1_k_array, D2_k_array)

    
#     res = []
#     for j in range(nPoints):        
#         res.append(x_data[0, j] - u_1_array[0, j])        
#         res.append(x_data[1, j] - u_1_array[1, j])        
#         res.append(x_data[0, j+nPoints] - u_2_array[0, j])
#         res.append(x_data[1, j+nPoints] - u_2_array[1, j])
        
#     for i in range(nPairs-1):
#         theta_rot = Op[i*6:i*6+3]
#         tras = Op[i*6+3:i*6+6]
#         T_wAwB = ObtainPose(theta_rot, tras[0], tras[1]) 
#         T_wAwB[0:3,3] = Op[i*6+3:i*6+6] 
#         x_3d_B = T_wAwB @ x_3dp
#         u_1_B_array, u_2_B_array = project_points_to_cameras_kannala(x_3d_B, T_c1_w, T_c2_w, K_1, K_2, D1_k_array, D2_k_array)
               
#         for j in range(nPoints):                 
#             res.append(x_data[0, j+2*i*nPoints+2*nPoints] - u_1_B_array[0, j])
#             res.append(x_data[1, j+2*i*nPoints+2*nPoints] - u_1_B_array[1, j])            
#             res.append(x_data[0, j+2*i*nPoints+2*nPoints+nPoints] - u_2_B_array[0, j])
#             res.append(x_data[1, j+2*i*nPoints+2*nPoints+nPoints] - u_2_B_array[1, j])
#     return res

# def resBundleProjection(Op, x_data, T_w_c1, T_w_c2, K_1, K_2, D1_k_array, D2_k_array, nPairs=2):
#     """
#     Compute residuals between observed 2D points and projected 3D points for bundle adjustment.

#     Input:
#         Op: Parameters to optimize (rotation, translation, 3D points).
#         x_data: Observed 2D points (shape: 2 x total_points).
#         T_w_c1: Pose of camera 1 in the world frame (4x4).
#         T_w_c2: Pose of camera 2 in the world frame (4x4).
#         K_1, K_2: Intrinsic calibration matrices for camera 1 and 2 (3x3).
#         D1_k_array, D2_k_array: Distortion coefficients for camera 1 and 2.
#         nPairs: Number of camera pairs.
    
#     Output:
#         res: Residuals (difference between observed and projected points).
#     """

#     def compute_residuals(observed, projected):
#         """Compute residuals between observed and projected 2D points."""
#         return (observed - projected).flatten()

#     def project_points_to_pair(T_c1_w, T_c2_w, X_3D, K_1, K_2, D1_k_array, D2_k_array):
#         """Project 3D points into camera 1 and camera 2."""
#         u_1, u_2 = project_points_to_cameras_kannala(X_3D, T_c1_w, T_c2_w, K_1, K_2, D1_k_array, D2_k_array)
#         return u_1, u_2

#     # --- Extract 3D Points and Camera Parameters ---
#     posStartX = 6 * (nPairs - 1)  # Start index for 3D points
#     nPoints = int((Op.shape[0] - posStartX) / 3)  # Number of 3D points
#     X_3D = Op[posStartX:].reshape(3, nPoints)  # Extract 3D points
#     X_3D_h = np.vstack([X_3D, np.ones((1, nPoints))])  # Homogeneous coordinates

#     # --- Inverse Camera Poses ---
#     T_c1_w = np.linalg.inv(T_w_c1)
#     T_c2_w = np.linalg.inv(T_w_c2)

#     # --- Compute Residuals for Base Camera Pair (Frame A) ---
#     u_A_c1, u_A_c2 = project_points_to_pair(T_c1_w, T_c2_w, X_3D_h, K_1, K_2, D1_k_array, D2_k_array)
#     res = []
#     res.extend(compute_residuals(x_data[:, :nPoints], u_A_c1[:2, :]))
#     res.extend(compute_residuals(x_data[:, nPoints:2 * nPoints], u_A_c2[:2, :]))

#     # --- Compute Residuals for Additional Camera Pairs ---
#     for i in range(nPairs - 1):
#         theta_rot = Op[i * 6:i * 6 + 3]  # Rotation parameters
#         tras = Op[i * 6 + 3:i * 6 + 6]  # Translation parameters

#         # Build transformation for this pair
#         T_wAwB = ObtainPose(theta_rot, tras[0], tras[1])
#         T_wAwB[0:3, 3] = tras

#         # Transform 3D points to this pair's frame
#         X_3D_B = T_wAwB @ X_3D_h

#         # Project points for this pair (Frame B)
#         u_B_c1, u_B_c2 = project_points_to_pair(T_c1_w, T_c2_w, X_3D_B, K_1, K_2, D1_k_array, D2_k_array)
#         offset = 2 * nPoints + 2 * i * nPoints
#         res.extend(compute_residuals(x_data[:, offset:offset + nPoints], u_B_c1[:2, :]))
#         res.extend(compute_residuals(x_data[:, offset + nPoints:offset + 2 * nPoints], u_B_c2[:2, :]))

#     return res


def project_points_to_cameras_kannala(X_3D, T_c1_w, T_c2_w, K_1, K_2, D1, D2):
    """Project 3D points onto the two cameras and return the projections."""

    # Transform 3D points to each camera's frame
    n_points = X_3D.shape[1]
    x_3d_c1= np.zeros((4, n_points))
    x_3d_c2= np.zeros((4, n_points))
    x_3d_c1 = T_c1_w @ X_3D
    x_3d_c2 = T_c2_w @ X_3D
    for i in range(n_points):
        x_3d = X_3D[:,i]
        x_3d_c1 [:,i] = T_c1_w @ x_3d
        x_3d_c2 [:,i] = T_c2_w @ x_3d
        
    # Project using the Kannala-Brandt model
    u_1_array = kannala_forward_model(x_3d_c1, K_1, D1)
    u_2_array = kannala_forward_model(x_3d_c2, K_2, D2)

    return u_1_array, u_2_array

def compute_residuals(x_observed, u_projected, nPoints):
    """Compute residuals between observed and projected points."""
    res = []
    for i in range(nPoints):
        res.append(x_observed[0, i] - u_projected[0, i])
        res.append(x_observed[1, i] - u_projected[1, i])
    return res

def exercise_2_1(K_1, D1_k_array):
        
        # EXERCICE 2.1.1 :Kannala-Brandt PROJECTION model #
        
        print("\n*******EXERCICE 2.1.1 :Kannala-Brandt PROJECTION model*******\n")
    
        x_1 = np.array([[3], [2], [10], [1]])
        x_2 = np.array([[-5], [6], [7], [1]])
        x_3 = np.array([[1], [5], [14], [1]])
        u_1 = kannala_forward_model(x_1, K_1, D1_k_array)
        u_2 = kannala_forward_model(x_2, K_1, D1_k_array)
        u_3 = kannala_forward_model(x_3, K_1, D1_k_array)
        
        print("u_1: ", u_1)
        print("u_2: ", u_2)
        print("u_3: ", u_3)

        # EXERCICE 2.1.1 :Kannala-Brandt UNPROJECTION model #
        
        print("\n*******EXERCICE 2.1.1 :Kannala-Brandt UNPROJECTION model*******\n")

        v_1 = kannala_backward_model(u_1, K_1, D1_k_array)
        v_2 = kannala_backward_model(u_2, K_1, D1_k_array)
        v_3 = kannala_backward_model(u_3, K_1, D1_k_array)

        print("\nv_1: ", v_1)
        print("x_1:", (x_1[0:3]/np.linalg.norm(x_1[0:3])).flatten())
        print("\nv_2: ", v_2)
        print("x_2:", (x_2[0:3]/np.linalg.norm(x_2[0:3])).flatten())
        print("\nv_3: ", v_3)
        print("x_3:", (x_3[0:3]/np.linalg.norm(x_3[0:3])).flatten())

def exercise_2_2(K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2, T_wAwB):
        
    print("\n*************EXERCICE 2.2: Triangulation*************\n")
    
    # Load points
    x1 = load_matrix("./data/x1.txt") # Position A, Camera 1
    x2 = load_matrix("./data/x2.txt") # Position A, Camera 2
    x3 = load_matrix("./data/x3.txt") # Position B, Camera 1
    x4 = load_matrix("./data/x4.txt") # Position B, Camera 2
    n_points = x1.shape[1]

    # Triangulate the points
    X_B_w = np.zeros((4, n_points))
    X_A_w = triangulation_kannala(x1, x2, K_1, K_2, D1_k_array, D2_k_array, T_w_c1=T_w_c1, T_w_c2=T_w_c2)
    X_B_w = transform_points(T_wAwB, X_A_w)

    #Plot the 3D 
    fig3D = plt.figure(1)
    ax = setup_3D_plot(x_label="X", y_label="Y", z_label="Z", equal_axis=True)
    drawRefSystem(ax, T_w_c1, "-", "L")
    drawRefSystem(ax, T_w_c2, "-", "R")
    plot_points_3D(ax, X_A_w, marker=".", color="b", size=10)
    plot_points_3D(ax, X_B_w, marker=".", color="r", size=10)
    print('IMAGE: 3D points in the 3D representation.')
    print('Close the image to continue...\n')
    plt.show(block=True)
        
    # Project the triangulated point to the camera 1
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)
    u_A_c1 = np.zeros((3, n_points))
    u_A_c2 = np.zeros((3, n_points))
    u_B_c1 = np.zeros((3, n_points))
    u_B_c2 = np.zeros((3, n_points))
    for point in range(n_points):
        x_A_c1 = T_c1_w @ X_A_w[:, point]
        x_A_c2 = T_c2_w @ X_A_w[:, point]
        x_B_c1 = T_c1_w @ T_wAwB @ X_A_w[:, point]
        x_B_c2 = T_c2_w @ T_wAwB @ X_A_w[:, point]
        u_A_c1[:, point] = kannala_forward_model(x_A_c1, K_1, D1_k_array).flatten()
        u_A_c2[:, point] = kannala_forward_model(x_A_c2, K_2, D2_k_array).flatten()
        u_B_c1[:, point] = kannala_forward_model(x_B_c1, K_1, D1_k_array).flatten()
        u_B_c2[:, point] = kannala_forward_model(x_B_c2, K_2, D2_k_array).flatten()
        
    # Draw the points in the images
    img1 = cv2.imread("./data/fisheye1_frameA.png")
    img2 = cv2.imread("./data/fisheye2_frameA.png")
    img3 = cv2.imread("./data/fisheye1_frameB.png")
    img4 = cv2.imread("./data/fisheye2_frameB.png")
    plot2DComparation(u_A_c1,x1,img1,"Camera 1 - Frame A",1)
    plot2DComparation(u_A_c2,x2,img2,"Camera 2 - Frame A",2)
    plot2DComparation(u_B_c1,x3,img3,"Camera 1 - Frame B",3)
    plot2DComparation(u_B_c2,x4,img4,"Camera 2 - Frame B",4)
    plt.show(block=True)  # Block execution until the figure is closed
    
    return X_A_w

def exercise_3(x_A_w, K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2, T_wAwB_seed):
    
    print("\n********EXERCICE 3: Bundle adjustment using calibrated stereo with fish-eyes*******\n")
    
    # Calculate the initial pose
    theta_rot, tras = Parametrice_Pose(T_wAwB_seed)
    tras = T_wAwB_seed[0:3,3]
    n_points = x_A_w.shape[1]
    
    # Construcción del Vector de Parámetros Optimizables 
    Op = np.hstack([
        np.array(theta_rot).flatten(),  # Parámetros de rotación (3 elementos)
        np.array(tras).flatten(),       # Parámetros de traslación (3 elementos)
        x_A_w[0:3,:].flatten()          # Coordenadas iniciales de los puntos 3D (apilados en un vector)
    ])
    
    # Load points
    x1 = load_matrix("./data/x1.txt") # Position A, Camera 1
    x2 = load_matrix("./data/x2.txt") # Position A, Camera 2
    x3 = load_matrix("./data/x3.txt") # Position B, Camera 1
    x4 = load_matrix("./data/x4.txt") # Position B, Camera 2
    n_points = x1.shape[1]
    x_data = np.hstack([x1, x2, x3, x4]) # Stack the points in a single array (before/after T_wAwB)

    OpOptim = scOptim.least_squares(
        resBundleProjection,
        Op,
        args=(x_data, T_w_c1, T_w_c2, K_1, K_2, D1_k_array, D2_k_array, 2)
    )

    #### Reconstrucción de Parámetros Optimizados ####
    # Reconstrucción de la matriz de transformación T_wAwB
    T_wAwB_optim = ObtainPose(OpOptim.x[0:3], OpOptim.x[3], OpOptim.x[4])
    T_wAwB_optim[0:3,3] = OpOptim.x[3:6]
    # Reconstrucción de las coordenadas 3D optimizadas
    x_A_w_optim = OpOptim.x[6:].reshape((3, int(OpOptim.x[6:].shape[0]/3)))
    x_A_w_optim = np.vstack([x_A_w_optim, np.ones((1, x_A_w_optim.shape[1]))])
    
    # Project the triangulated point to the camera 1
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)
    x_B_w_optim = np.zeros((4, n_points))
    u_A_c1 = np.zeros((3, n_points))
    u_A_c2 = np.zeros((3, n_points))
    u_B_c1 = np.zeros((3, n_points))
    u_B_c2 = np.zeros((3, n_points))
    for point in range(n_points):
        x_A_c1 = T_c1_w @ x_A_w_optim[:, point]
        x_A_c2 = T_c2_w @ x_A_w_optim[:, point]
        x_B_w_optim[:, point] = T_wAwB_optim @ x_A_w_optim[:, point]
        x_B_c1 = T_c1_w @ x_B_w_optim[:, point]
        x_B_c2 = T_c2_w @ x_B_w_optim[:, point]
        u_A_c1[:, point] = kannala_forward_model(x_A_c1, K_1, D1_k_array).flatten()
        u_A_c2[:, point] = kannala_forward_model(x_A_c2, K_2, D2_k_array).flatten()
        u_B_c1[:, point] = kannala_forward_model(x_B_c1, K_1, D1_k_array).flatten()
        u_B_c2[:, point] = kannala_forward_model(x_B_c2, K_2, D2_k_array).flatten()


    # Draw the points in the images
    img1 = cv2.imread("./data/fisheye1_frameA.png")
    img2 = cv2.imread("./data/fisheye2_frameA.png")
    img3 = cv2.imread("./data/fisheye1_frameB.png")
    img4 = cv2.imread("./data/fisheye2_frameB.png")
    plot2DComparation(u_A_c1,x1,img1,"Camera 1 - Frame A",1)
    plot2DComparation(u_A_c2,x2,img2,"Camera 2 - Frame A",2)
    plot2DComparation(u_B_c1,x3,img3,"Camera 1 - Frame B",3)
    plot2DComparation(u_B_c2,x4,img4,"Camera 2 - Frame B",4)
    plt.show(block=True)  # Block execution until the figure is closed
    
    #Plot the 3D 
    fig3D = plt.figure(10)
    ax = setup_3D_plot(x_label="X", y_label="Y", z_label="Z", equal_axis=True)
    drawRefSystem(ax, T_w_c1, "-", "L")
    drawRefSystem(ax, T_w_c2, "-", "R")
    plot_points_3D(ax, x_A_w_optim, marker=".", color="b", size=10)
    plot_points_3D(ax, x_B_w_optim, marker=".", color="r", size=10)
    print('IMAGE: 3D points in the 3D representation.')
    print('Close the image to continue...\n')
    plt.show(block=True)
    
    return None

def main():
        
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # Load matrices
    K_1 = load_matrix("./data/K_1.txt")
    K_2 = load_matrix("./data/K_2.txt")    
    D1_k_array = load_matrix("./data/D1_k_array.txt")
    D2_k_array = load_matrix("./data/D2_k_array.txt")
    T_w_c1 = load_matrix("./data/T_wc1.txt")
    T_w_c2 = load_matrix("./data/T_wc2.txt")
    T_wAwB_gt = load_matrix("./data/T_wAwB_gt.txt")
    T_wAwB_seed = load_matrix("./data/T_wAwB_seed.txt")
    
    ### EXERCICE 2.1: Kannala-Brandt Model ###
    
    if EXERCICE_2_1:
        exercise_2_1(K_1, D1_k_array)

    ### EXERCICE 2.2: Triangulation ###
    
    if EXERCICE_2_2:
        x_A_w = exercise_2_2(K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2, T_wAwB_gt)
     
    ### EXERCICE 3: Bundle adjustment ###
    
    if EXERCICE_3:
        exercise_3(x_A_w, K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2, T_wAwB_seed)
                     

### Exercices flags ###
EXERCICE_2_1 = True
EXERCICE_2_2 = True
EXERCICE_3 = True
DEBUG = True

if __name__ == "__main__":
    
    main()
    