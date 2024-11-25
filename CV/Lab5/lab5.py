import numpy as np
import os
import matplotlib.pyplot as plt
import cv2

from utils.drawingCV import *


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
    A = np.vstack((d_theta * np.cos(phi), d_theta * np.sin(phi), 1))
    return Kc @ A 

def kannala_forward_model(x, K_c, D_k):
    
    # Ensure the input vectors are column vectors
    x = ensure_column_vector(x)
    
    # Step 1: Compute the radial distance
    r_dist = compute_radial_distance(x)
    
    # Step 2: Compute the theta angle
    theta = np.arctan2(r_dist[0], x[2])
    phi = np.arctan2(x[1], x[0])  # Ángulo azimutal

    # Step 3: Calcule la distorsion radiale
    d_theta = compute_distorsion_radiale(theta, D_k)

    # Step 4: Compute the projection
    u = compute_Kannala_Brandt_projection(K_c, d_theta, phi)
    
    return u.flatten()

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
    
# TODO: Implementar varios calculos     
# assert v1.shape[1] == v2.shape[1], "v1 and v2 must have the same number of points"
def triangulation_kannala(v1, v2, T_w_c1, T_w_c2):
    """
    Realiza la triangulación de un punto a partir de dos rayos en las cámaras 1 y 2.
    Parámetros:
    - v1: Rayo en el marco de la cámara 1 (3x1 columna).
    - v2: Rayo en el marco de la cámara 2 (3x1 columna).
    - T_c1_w: Transformación de la cámara 1 al marco del mundo (4x4).
    - T_c2_w: Transformación de la cámara 2 al marco del mundo (4x4).
    Retorna:
    - X_cam1: Coordenadas trianguladas en el marco de la cámara 1 (4x1).
    """
    


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
    X_tri_FRW = T_w_c2 @ X_tri_FR2
    debug_print("X_tri_FRW: ", X_tri_FRW, "\n")
    
    return X_tri_FRW


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

def exercise_2_2(K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2):
        
        print("\n*************EXERCICE 2.2: Triangulation*************\n")
        
        # Load points
        x1 = load_matrix("./data/x1.txt") # Position A, Camera 1
        x2 = load_matrix("./data/x2.txt") # Position A, Camera 2
        x3 = load_matrix("./data/x3.txt") # Position B, Camera 1
        x4 = load_matrix("./data/x4.txt") # Position B, Camera 2
        n_points = x1.shape[1]

        # Triangulate the points
        X_A_w = np.zeros((4, n_points))
        for i in range(n_points):
            v1 = kannala_backward_model(x1[:, i], K_1, D1_k_array)
            v2 = kannala_backward_model(x2[:, i], K_2, D2_k_array)
            debug_print("\nX1: ", x1[:, i])
            debug_print("X2: ", x2[:, i])
            debug_print("V1: ", v1)
            debug_print("V2: ", v2)
            X_A_w[:, i] = triangulation_kannala(v1, v2, T_w_c1=T_w_c1, T_w_c2=T_w_c2)
            print(f"Point {i}: 3D Coord: {X_A_w[:, i].flatten()}")
            
        #Plot the 3D 
        fig3D = plt.figure(1)
        ax = setup_3D_plot(x_label="X", y_label="Y", z_label="Z", equal_axis=True)
        drawRefSystem(ax, T_w_c1, "-", "L")
        drawRefSystem(ax, T_w_c2, "-", "R")
        plot_points_3D(ax, X_A_w, marker=".", color="b", size=10)
        print('\nClick in the image to continue...\n')
        plt.show(block=True)
            
        # Project the triangulated point to the camera 1
        u_tri_c1 = np.zeros((3, n_points))
        T_c1_w = np.linalg.inv(T_w_c1)
        T_c2_w = np.linalg.inv(T_w_c2)
    
        for point in range(n_points):
            x_c1 = T_c1_w @ X_A_w[:, point]
            u_tri_c1[:, point] = kannala_forward_model(x_c1, K_1, D1_k_array).flatten()
            print(f"Point {point} - Coord projected: {u_tri_c1[:, point]}")
            
        # Check the triangulation results
        img1 = cv2.imread("./data/fisheye1_frameA.png")
        img2 = cv2.imread("./data/fisheye1_frameB.png")
        img3 = cv2.imread("./data/fisheye2_frameA.png")
        img4 = cv2.imread("./data/fisheye2_frameB.png")

        plt.figure()
        plt.imshow(img1)
        plt.scatter(x1[0], x1[1], c='r', marker='x')
        plt.scatter(u_tri_c1[0, :], u_tri_c1[1, :], c='b', marker='o', s=6)
        plt.show(block=True)  # Block execution until the figure is closed


def main():
        
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # Load matrices
    K_1 = load_matrix("./data/K_1.txt")
    K_2 = load_matrix("./data/K_2.txt")    
    D1_k_array = load_matrix("./data/D1_k_array.txt")
    D2_k_array = load_matrix("./data/D2_k_array.txt")
    T_w_c1 = load_matrix("./data/T_wc1.txt")
    T_w_c2 = load_matrix("./data/T_wc2.txt")
    
    ### EXERCICE 2.1: Kannala-Brandt Model ###
    
    if EXERCICE_2_1:
        exercise_2_1(K_1, D1_k_array)

    # EXERCICE 2.2: Triangulation #
    
    if EXERCICE_2_2:
        exercise_2_2(K_1, K_2, D1_k_array, D2_k_array, T_w_c1, T_w_c2)
             

### Exercices flags ###
EXERCICE_2_1 = True
EXERCICE_2_2 = True
DEBUG = False

if __name__ == "__main__":
    
    main()
    
    # # main()
    # x1 = np.array([[4], [0], [10], [1]])
    # K = np.array([[300, 0, 400], [0, 300, 400], [0, 0, 1]])
    # D1 = np.array([0, 0.01, 0, 0, 0])
    # u1 = kannala_forward_model(x1, K, D1)
    # print(u1)
    
    # v_1 = kannala_backward_model(u1, K, D1)
    # print(v_1)
    
    # v_1 = np.array([[0.3717], [0], [0.9285]])
    # v_2 = np.array([[-8], [0], [10]])
    # v_2 = v_2 / np.linalg.norm(v_2)
    
    # T_w_c1 = np.array([[1, 0, 0, 6], [0, 1, 0, 0], [0, 0, 1, 0  ], [0, 0, 0, 1]])
    # T_w_c2 = np.array([[1, 0, 0, 6], [0, 1, 0, 0], [0, 0, 1, 10], [0, 0, 0, 1]])
    
    # X_A_w = triangulation_kannala(v_1, v_2,T_w_c1,T_w_c1)
    # print(X_A_w)


        #     for point in range(n_points):
        #     x_A_3D[:, point] = triangulation_kannala(x1[:, point], x2[:, point], K_1, K_2, D1_k_array, D2_k_array, T_wc1, T_wc2)
        #     print(f"Point {point}: 3D Coord: {x_A_3D[:, point]}")
        
        # # Re-projection of the triangulated points
        # x_cam1 = T_wc1 @ x_A_3D
        # n_points = x_cam1.shape[1]
        # u_tri_x1 = np.zeros((3, n_points))
        # for point in range(n_points):
        #     u_tri_x1[:, point] = kannala_forward_model(x_cam1[:, point], K_1, D1_k_array).flatten()
        #     print(f"Point {point}: 3D Coord: {u_tri_x1[:, point]}")
        # u_tri_x1[0, :] /= u_tri_x1[2, :]
        # u_tri_x1[1, :] /= u_tri_x1[2, :]