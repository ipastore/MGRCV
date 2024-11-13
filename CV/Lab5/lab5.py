import numpy as np
import os

def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {str(e)}")


def compute_radial_distance(points):
    return np.sqrt(points[0]**2 + points[1]**2)

def compute_distorsion_radiale(theta, D_k_array):
    d_theta = theta + theta**3 * D_k_array[0] + theta**5 * D_k_array[1] + theta**7 * D_k_array[2] + theta**9 * D_k_array[3]
    return d_theta

def compute_Kannala_Brandt_projection(Kc, d_theta, phi):
    A = np.vstack((d_theta * np.cos(phi), d_theta * np.sin(phi), 1))
    return Kc @ A 

def kannala_forward_model(x, K_c, D_k):
    
    # Step 1: Compute the radial distance
    r_dist = compute_radial_distance(x)
    
    # Step 2: Compute the theta angle
    theta = np.arctan2(r_dist[0], x[2])
    phi = np.arctan2(x[1], x[0])  # Ángulo azimutal

    # Step 3: Calcule la distorsion radiale
    d_theta = compute_distorsion_radiale(theta, D_k)

    # Step 4: Compute the projection
    u = compute_Kannala_Brandt_projection(K_c, d_theta, phi)
    
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
    # Step 1: Normalize homogeneous coordinates
    # u = u / u[2, 0]  # Ensure u[2] == 1 for homogeneous coordinates
    
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
    
    print(theta)
    
    # Step 6: Compute 3D ray in camera coordinates
    sin_theta = np.sin(theta)
    cos_theta = np.cos(theta)
    cos_phi = np.cos(phi)
    sin_phi = np.sin(phi)
    
    # Construct the ray as a (3, 1) column vector
    v = np.array([
        [sin_theta * cos_phi],
        [sin_theta * sin_phi],
        [cos_theta]
    ])
    
    return v

def triangulation_kannala(x1, x3, K_1, D1_k_array, T_wc1, T_wc2):
    
    n_points = x1.shape[1]
    v1 = np.zeros((3, n_points))
    v2 = np.zeros((3, n_points))
    
    for point in range(n_points):
        u1 = x1[:, point]
        u2 = x3[:, point]
        v1[:, point] = kannala_backward_model(u1, K_1, D1_k_array)
        v2[:, point] = kannala_backward_model(u2, K_1, D1_k_array)
    
    # Compute the 3D points
    x_A_3D = np.zeros((4, n_points))
    for i in range(n_points): 
        ray_1 = v_1[:, i]
        ray_2 = v_2[:, i]
        
    
    return None

if __name__ == "__main__":

    
    ### Provided data ###
    
    # Load matrices
    K_1 = load_matrix("./data/K_1.txt")
    K_2 = load_matrix("./data/K_2.txt")    
    D1_k_array = load_matrix("./data/D1_k_array.txt")
    D2_k_array = load_matrix("./data/D2_k_array.txt")
    T_wc1 = load_matrix("./data/T_wc1.txt")
    T_wc2 = load_matrix("./data/T_wc2.txt")
    
    ### EXERCICE 2.1: Kannala-Brandt Model ###
    
    # EXERCICE 2.1.1 :Kannala-Brandt PROJECTION model #
    
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
    
    v_1 = kannala_backward_model(u_1, K_1, D1_k_array)
    v_2 = kannala_backward_model(u_2, K_1, D1_k_array)
    v_3 = kannala_backward_model(u_3, K_1, D1_k_array)

    print("\nv_1: ", v_1)
    print("\nx_1:", x_1[0:3]/np.linalg.norm(x_1[0:3]), sep="\n")
    print("\nv_2: ", v_2)
    print("\nx_2:", x_2[0:3]/np.linalg.norm(x_2[0:3]), sep="\n")
    print("\nv_3: ", v_3)
    print("\nx_3:", x_3[0:3]/np.linalg.norm(x_3[0:3]), sep="\n")

    
    # EXERCICE 2.2: Triangulation #
    
    # Load points
    x1 = load_matrix("./data/x1.txt") # Position A, Camera 1
    x2 = load_matrix("./data/x2.txt") # Position B, Camera 1
    x3 = load_matrix("./data/x3.txt") # Position A, Camera 2
    x4 = load_matrix("./data/x4.txt") # Position B, Camera 2
    
    x_A_3D = triangulation_kannala(x1, x3, K_1, D1_k_array, T_wc1, T_wc2)
    
    
    
    