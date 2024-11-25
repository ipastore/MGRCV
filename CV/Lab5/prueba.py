import numpy as np
import os
import matplotlib.pyplot as plt
import cv2


def kannala_triangularization_v2(x1, x2, K_1, K_2, D1_k_array, D2_k_array, T_1_2, T_w_2 = np.eye(4)):
    """
    This function implements the triangulation algorithm based on planes.
    Inputs:
        x1: points in camera 1
        x2: points in camera 2
    Outputs:
        p: points in 3D
    """
    assert x1.shape[1] == x2.shape[1]   # to ensure both x1 and x2 have the same number of points
    n_points = x1.shape[1]
    v_1 = np.empty([3, n_points])
    v_2 = np.empty([3, n_points])
    for i in range(n_points):
        # append to the np array
        point_1 = x1[:, i]
        v_1[:, i] = kannala_backward_model_v2(point_1, K_1, D1_k_array)
        point_2 = x2[:, i]
        v_2[:, i] = kannala_backward_model_v2(point_2, K_2, D2_k_array)
    
    x_3d_array = np.empty([4, n_points])
    for i in range(n_points):
        ray_1 = v_1[:, i]
        ray_2 = v_2[:, i]
        plane_sym_1 = np.array([-ray_1[1], ray_1[0], 0, 0])
        plane_perp_1 = np.array([-ray_1[2]*ray_1[0], -ray_1[2]*ray_1[1], ray_1[0]**2 + ray_1[1]**2, 0])
        plane_sym_2 = np.array([-ray_2[1], ray_2[0], 0, 0])
        plane_perp_2 = np.array([-ray_2[2]*ray_2[0], -ray_2[2]*ray_2[1], ray_2[0]**2 + ray_2[1]**2, 0])
        
        plane_sym_1_2 = T_1_2.T @ plane_sym_1
        plane_perp_1_2 = T_1_2.T @ plane_perp_1
        A = np.array([plane_sym_1_2.T, plane_perp_1_2.T, plane_sym_2.T, plane_perp_2.T])
        u, s, vh = np.linalg.svd(A)
        # ensure rank 3 for A
        S = np.zeros([4, 4])
        S[0, 0] = s[0]
        S[1, 1] = s[1]
        S[2, 2] = s[2]
        A = u @ S @ vh
        u, s, vh = np.linalg.svd(A)
        # now we can get the 3d point
        x_3d_cam2 = vh[-1, :]
        # now bring the point to the world frame
        x_3d = T_w_2 @ x_3d_cam2
        x_3d /= x_3d[3]
        x_3d_array[:, i] = x_3d
    return x_3d_array


def kannala_backward_model_v2(u, K_c, D):
    """
    This function implements the Kannala-Brandt unprojection model.
    Inputs:
        u: 2d coordinates on the image
        K_c: camera calibration matrix
        D: distortion coefficients
    Outputs:
        v: ray in the camera frame
    """
    u = np.array(u, dtype=np.float64)
    u /= u[2]   # sanity check
    x_1 = np.linalg.inv(K_c) @ u
    d = np.sqrt((x_1[0]**2 + x_1[1]**2)/(x_1[2]**2))
    phi = np.arctan2(x_1[1], x_1[0])
    theta_solutions = np.roots([D[3], 0, D[2], 0, D[1], 0, D[0], 0, 1, -d])
    # get the real root
    for theta_value in theta_solutions:
        if theta_value.imag == 0:
            theta = theta_value.real
            break
    v = np.array([np.sin(theta)*np.cos(phi), np.sin(theta)*np.sin(phi), np.cos(theta)])
    return v

def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {str(e)}")


def ensure_column_vector(array):
    return array.reshape(-1, 1) if array.ndim == 1 else array

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
    
    v = np.array([
        sin_theta * cos_phi,
        sin_theta * sin_phi,
        cos_theta
    ])
    
    return v

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


# Load matrices
K_1 = load_matrix("./data/K_1.txt")
K_2 = load_matrix("./data/K_2.txt")    
D1_k_array = load_matrix("./data/D1_k_array.txt")
D2_k_array = load_matrix("./data/D2_k_array.txt")
T_w_c1 = load_matrix("./data/T_wc1.txt")
T_w_c2 = load_matrix("./data/T_wc2.txt")
   
# Load points
# x1 = load_matrix("./data/x1.txt") # Position A, Camera 1
# x2 = load_matrix("./data/x2.txt") # Position A, Camera 2


x_1 = np.array([[3], [2], [10], [1]])
x_2 = np.array([[-5], [6], [7], [1]])
x_3 = np.array([[1], [5], [14], [1]])
u_1 = kannala_forward_model(x_1, K_1, D1_k_array)
u_2 = kannala_forward_model(x_2, K_1, D1_k_array)
u_3 = kannala_forward_model(x_3, K_1, D1_k_array)

print("u_1: ", u_1)
print("u_2: ", u_2)
print("u_3: ", u_3)

# Get vectors from the points

v1_1 = kannala_backward_model(u_1, K_1, D1_k_array)
v1_2 = kannala_backward_model(u_2, K_2, D2_k_array)
v1_1 = kannala_backward_model_v2(u_1, K_1, D1_k_array)
v1_2 = kannala_backward_model_v2(u_2, K_2, D2_k_array)
print("\nX1: ", u_1)
print("V1_v1: ", v1_1)
print("V1_v2: ", v1_2)
print("x_1:", (x_1[0:3]/np.linalg.norm(x_1[0:3])).flatten())

print("\nX2: ", u_2)
print("V2_v1: ", v1_1)
print("V2_v2: ", v1_2)
print("x_2:", (x_1[0:3]/np.linalg.norm(x_1[0:3])).flatten())


## Triangulation
# v1_1 = kannala_backward_model(x1[:, 1], K_1, D1_k_array)
# v1_2 = kannala_backward_model(x2[:, 1], K_2, D2_k_array)
# v1_1 = kannala_backward_model_v2(x1[:, 1], K_1, D1_k_array)
# v1_2 = kannala_backward_model_v2(x2[:, 1], K_2, D2_k_array)
# print("\nX1: ", x1[:, 1])
# print("V1_v1: ", v1_1)
# print("V1_v2: ", v1_2)
# print("\nX2: ", x2[:, 1])
# print("V2_v1: ", v1_1)
# print("V2_v2: ", v1_2)
