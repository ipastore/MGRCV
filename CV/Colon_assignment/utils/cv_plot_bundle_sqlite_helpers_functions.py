import numpy as np
import json 
import sqlite3
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import csv
import scipy as sc
import scipy.io as sio
import cv2
import sqlite3
from collections import defaultdict


#################################################### CV FUNCTIONS ####################################################
#region CV FUNCTIONS

def compute_essential_matrix_from_F(F, K1, K2):
    """
        Compute the essential matrix from the fundamental matrix and intrinsic matrices of the two cameras.
        - Input:
            · F (np.array): Fundamental matrix (3x3)
            · K1, K2 (np.array): Intrinsic matrices of the two cameras (3x3)
        - Output:
            · E (np.array): Essential matrix (3x3) 
    """
    return K2.T @ F @ K1

def decompose_essential_matrix(E):
    """
        Decompose the essential matrix into rotation and translation.
        - Input:
            · E (np.array): Essential matrix (3x3)
        - Output:
            · R1, R2: Possible rotation matrices (3x3)
            · t: Translation vector
    """
    U, _, Vt = np.linalg.svd(E)

    if np.linalg.det(U) < 0:
        U *= -1
    if np.linalg.det(Vt) < 0:
        Vt *= -1

    W = np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]])
    R1 = U @ W @ Vt
    R2 = U @ W.T @ Vt
    t = U[:, 2]

    return R1, R2, t

def is_valid_pose(X, P1, P2):
    """
        Check if the triangulated 3D points are in front of both cameras.
        - Input:
            · X (np.array): Triangulated 3D points (4xN, homogeneous coordinates)
            · P1, P2 (np.array): Projection matrices (3x4)
        - Output:
            · bool: True if the points are in front of both cameras, False otherwise
    """
    depth1 = P1[2, :] @ X
    depth2 = P2[2, :] @ X

    return np.all(depth1 > 1e-6) and np.all(depth2 > 0)

def count_valid_points(X, P1, P2):
    """
     Count the number of valid points in front of both cameras.
     - Input:
         · X (np.array): Triangulated 3D points (4xN, homogeneous coordinates)
         · P1, P2 (np.array): Projection matrices (3x4)
        - Output:
            · int: Number of valid points
    """
    depth1 = P1[2, :] @ X
    depth2 = P2[2, :] @ X
    valid_depths1 = depth1 > 1e-6
    valid_depths2 = depth2 > 1e-6

    count_valid_depths1 = np.sum(valid_depths1)
    count_valid_depths2 = np.sum(valid_depths2)
    count_all_valid_depths = count_valid_depths1 + count_valid_depths2

    return count_all_valid_depths, valid_depths1, valid_depths2
    
def select_correct_pose_harsh(x1_h, x2_h, K1, K2, R1, R2, t):
    """
        Select the correct pose from the four possible solutions.
        - Input:
            · x1_h, x2_h (np.array): 2D points from image 1 and 2 (3xN, homogeneous coordinates)
            · K1, K2 (np.array): Intrinsic matrices of the two cameras (3x3)
            · R1, R2 (np.array): Possible rotation matrices (3x3)
            · t (np.array): Translation vector (3x1)
        - Output:
            · R, t: Correct rotation and translation
            · X: Triangulated 3D points (4xN, homogeneous coordinates)
    """

    R1_t = np.hstack((R1, t.reshape(3, 1)))
    R1_minust = np.hstack((R1, -t.reshape(3, 1)))
    R2_t = np.hstack((R2, t.reshape(3, 1)))
    R2_minust = np.hstack((R2, -t.reshape(3, 1)))

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2_1 = K2 @ R1_t
    P2_2 = K2 @ R1_minust
    P2_3 = K2 @ R2_t
    P2_4 = K2 @ R2_minust

    X1 = triangulate_points(x1_h, x2_h, P1, P2_1)
    X2 = triangulate_points(x1_h, x2_h, P1, P2_2)
    X3 = triangulate_points(x1_h, x2_h, P1, P2_3)
    X4 = triangulate_points(x1_h, x2_h, P1, P2_4)

    ##Plot the 3D cameras and the 3D points
    #Ensemble T
    T1 = ensamble_T(R1, t)
    T2 = ensamble_T(R1, -t)
    T3 = ensamble_T(R2, t)
    T4 = ensamble_T(R2, -t)

    # Transpose T
    T1 = np.linalg.inv(T1)
    T2 = np.linalg.inv(T2)
    T3 = np.linalg.inv(T3)
    T4 = np.linalg.inv(T4)
    

    ax = plt.axes(projection='3d', adjustable='box')
    fig3D = draw_possible_poses(ax, T1, X1)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_t')
    plt.show()
    #P2_2
    fig3D = draw_possible_poses(ax, T2, X2)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_minust')
    plt.show()
    #P2_3
    fig3D = draw_possible_poses(ax, T3, X3)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_t')
    plt.show()
    #P2_4
    fig3D = draw_possible_poses(ax, T4, X4)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_minust')
    plt.show()

    possible_P2_1 = is_valid_pose(X1, P1, P2_1)
    possible_P2_2 = is_valid_pose(X2, P1, P2_2)
    possible_P2_3 = is_valid_pose(X3, P1, P2_3)
    possible_P2_4 = is_valid_pose(X4, P1, P2_4)

    if possible_P2_1:
        print("Pose 1")
        return R1, t, X1
    elif possible_P2_2:
        print("Pose 2")
        return R1, -t, X2
    elif possible_P2_3:
        print("Pose 3")
        return R2, t, X3
    elif possible_P2_4:
        print("Pose 4")
        return R2, -t, X4
    else:
        print("No valid pose found, try with a different method")
        return None, None, None

def select_correct_pose_flexible_and_filter(x1_h, x2_h, K1, K2, R1, R2, t, match_list=None, plot_FLAG = False, filtering=True):
    """
    Select the best pose from the four possible solutions.
    """

    R1_t = np.hstack((R1, t.reshape(3, 1)))
    R1_minust = np.hstack((R1, -t.reshape(3, 1)))
    R2_t = np.hstack((R2, t.reshape(3, 1)))
    R2_minust = np.hstack((R2, -t.reshape(3, 1)))

    P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2_1 = K2 @ R1_t
    P2_2 = K2 @ R1_minust
    P2_3 = K2 @ R2_t
    P2_4 = K2 @ R2_minust

    # Triangulate points for each pose
    #Camera 1
    X1 = triangulate_points(x1_h, x2_h, P1, P2_1)
    X2 = triangulate_points(x1_h, x2_h, P1, P2_2)
    X3 = triangulate_points(x1_h, x2_h, P1, P2_3)
    X4 = triangulate_points(x1_h, x2_h, P1, P2_4)

    ##Plot the 3D cameras and the 3D points
    #Ensemble T
    T1 = ensamble_T(R1, t)
    T2 = ensamble_T(R1, -t)
    T3 = ensamble_T(R2, t)
    T4 = ensamble_T(R2, -t)

    # Transpose T
    T1 = np.linalg.inv(T1)
    T2 = np.linalg.inv(T2)
    T3 = np.linalg.inv(T3)
    T4 = np.linalg.inv(T4)
    
    
    if plot_FLAG:
        ##Plot the 3D cameras and the 3D points
        #P2_1
        ax = plt.axes(projection='3d', adjustable='box')
        fig3D = draw_possible_poses(ax, T1, X1)
        adjust_plot_limits(ax, X1)
        plt.title('Possible Pose R1_t')
        plt.show()
        #P2_2
        fig3D = draw_possible_poses(ax, T2, X2)
        adjust_plot_limits(ax, X1)
        plt.title('Possible Pose R1_minust')
        plt.show()
        #P2_3
        fig3D = draw_possible_poses(ax, T3, X3)
        adjust_plot_limits(ax, X1)
        plt.title('Possible Pose R2_t')
        plt.show()
        #P2_4
        fig3D = draw_possible_poses(ax, T4, X4)
        adjust_plot_limits(ax, X1)
        plt.title('Possible Pose R2_minust')
        plt.show()


    # Count valid points for each pose
    # BUG: Check why depths in camera2 are negative
    count1, valid_depths1X1, validdepths2X1 = count_valid_points(X1, P1, P2_1)
    count2, valid_depths1X2, validdepths2X2 = count_valid_points(X2, P1, P2_2)
    count3, valid_depths1X3, validdepths2X3 = count_valid_points(X3, P1, P2_3)
    count4, valid_depths1X4, validdepths2X4 = count_valid_points(X4, P1, P2_4)

    valid_counts = [count1, count2, count3, count4]
    valid_depths1 = [valid_depths1X1, valid_depths1X2, valid_depths1X3, valid_depths1X4]
    valid_depths2 = [validdepths2X1, validdepths2X2, validdepths2X3, validdepths2X4]
    X_list = [X1, X2, X3, X4]

    print("Valid counts")
    print(valid_counts)
    # Select the pose with the maximum valid points
    best_pose_idx = np.argmax(valid_counts)

    X = X_list[best_pose_idx]
    
    # # #DEBUG:
    # best_pose_idx = 1
    if filtering : 
        # For best pose id filter X x1_h and x2_h
        X_filtered = X[:, valid_depths1[best_pose_idx] & valid_depths2[best_pose_idx]]
        x1_h = x1_h[:, valid_depths1[best_pose_idx] & valid_depths2[best_pose_idx]]
        x2_h = x2_h[:, valid_depths1[best_pose_idx] & valid_depths2[best_pose_idx]]
        match_list = [match_list[i] for i in range(len(match_list)) if valid_depths1[best_pose_idx][i] & valid_depths2[best_pose_idx][i]]

        poses = [(R1, t, X_filtered, x1_h, x2_h, match_list), (R1, -t, X_filtered,x1_h, x2_h,match_list),
              (R2, t, X_filtered,x1_h, x2_h,match_list), (R2, -t, X_filtered,x1_h, x2_h,match_list)]
    else:
        poses = [(R1, t, X, x1_h, x2_h, match_list), (R1, -t, X, x1_h, x2_h, match_list),
              (R2, t, X, x1_h, x2_h, match_list), (R2, -t, X, x1_h, x2_h, match_list)]

    if valid_counts[best_pose_idx] == 0:
        raise ValueError("No valid pose found!")
    print("Best_pose_idx")
    print(best_pose_idx)
    if filtering:
        print(f"Points filtered: {X.shape[1] - X_filtered.shape[1]} with negative depth, out of {X.shape[1]}")
    return poses[best_pose_idx]  # Return the best pose (R, t, X)

def crossMatrix(x):
    """Creates a skew-symmetric cross-product matrix from a vector x."""
    M = np.array([
        [0, -x[2], x[1]],
        [x[2], 0, -x[0]],
        [-x[1], x[0], 0]
    ], dtype="object")
    return M

def crossMatrixInv(M):
    """Extracts a vector x from a skew-symmetric matrix M."""
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return np.array(x)

def project_to_camera(P, X_h):
    """Project homogeneous 3D points X_h to 2D using projection matrix P."""
    # Ensure X_h is in the shape (4, nPoints)
    if X_h.shape[0] != 4:
        raise ValueError(f"Expected X_h with shape (4, nPoints), but got shape {X_h.shape}")
    
    # Ensure P is in the shape (3, 4)
    if P.shape != (3, 4):
        raise ValueError(f"Expected P with shape (3, 4), but got shape {P.shape}")
    
    x_h_projected = P @ X_h
    x_h_projected /= x_h_projected[2, :]  # Normalize by the third coordinate
    
    return x_h_projected

def indexMatrixToMatchesList(matchesList):
    """
    Convert a numpy matrix of index in a list of DMatch OpenCv matches.
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0].astype('int'), _trainIdx=row[1].astype('int'), _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
    Convert a list of DMatch OpenCv matches into a numpy matrix of index.

     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([np.int(dMatchesList[k].queryIdx), np.int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def get_projection_matrix(K, T):
    """
    Computes the projection matrix (3x4) from camera matrix (K) and transformation matrix (T).
    - Inputs:
        · K (np.array): Camera intrinsic matrix (3x3).
        · T (np.array): Transformation matrix (4x4).
    - Output:
        · np.array: Projection matrix.
    """
    if T.shape != (4, 4):
        raise ValueError(f"Expected T with shape (4, 4), but got shape {T.shape}")
    
    Rt = T[:3, :]  # Extract the top 3 rows (3x4) from the 4x4 transformation matrix
    P = K @ Rt     # Multiply by the intrinsic matrix to get a 3x4 projection matrix
    return P

def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c.flatten()
    T_w_c[3, 3] = 1
    return T_w_c

def triangulate_points(x1, x2, P1, P2):
    """
        Triangulate 3D points from two sets of 2D correspondences, where points are stored in columns.
        - Inputs:
            · x1, x2 (np.array): 2D points from image 1 and 2 (shape: 2 x num_points, where each column is a point)
            · P1, P2 (np.array): Projection matrices (3x3) for camera 1 and 2
            
        - Output:
            · np.array: Triangulated points.
    """
    num_points = x1.shape[1]  # Number of points = the number of columns
    X_h = np.zeros((4, num_points))  # Converting it to homogeneous coordinates

    for i in range(num_points):
        A = np.zeros((4, 4))
        A[0] = x1[0, i] * P1[2] - P1[0]  # x1[0, i] is the x-coordinate of the i-th point
        A[1] = x1[1, i] * P1[2] - P1[1]  # x1[1, i] is the y-coordinate of the i-th point
        A[2] = x2[0, i] * P2[2] - P2[0]  # x2[0, i] is the x-coordinate of the i-th point in image 2
        A[3] = x2[1, i] * P2[2] - P2[1]  # x2[1, i] is the y-coordinate of the i-th point in image 2
        
        _, _, Vt = np.linalg.svd(A)
        X_h[:, i] = Vt[-1]
        X_h[:, i] /= X_h[-1, i]  # Normalization of homogeneous coordinates 

    return X_h

def validate_fundamental_matrix(F, matched_points1, matched_points2):
    """
    Validate whether the fundamental matrix F corresponds to F21 or F12.
    - F: Fundamental matrix (3x3)
    - matched_points1: Matched points from Image 1 (Nx2)
    - matched_points2: Matched points from Image 2 (Nx2)

    Returns:
        - "F21" if F maps points from Image 1 to lines in Image 2.
        - "F12" if F maps points from Image 2 to lines in Image 1.
    """
    matched_points1 = np.hstack((matched_points1, np.ones((matched_points1.shape[0], 1))))  # Convert to homogeneous
    matched_points2 = np.hstack((matched_points2, np.ones((matched_points2.shape[0], 1))))  # Convert to homogeneous

    # Compute epipolar lines
    lines_in_img2 = (F @ matched_points1.T).T  # Lines in Image 2 corresponding to points in Image 1
    lines_in_img1 = (F.T @ matched_points2.T).T  # Lines in Image 1 corresponding to points in Image 2

    # Compute distances to the epipolar lines
    distances_img2 = np.abs(np.sum(lines_in_img2 * matched_points2, axis=1)) / np.sqrt(lines_in_img2[:, 0]**2 + lines_in_img2[:, 1]**2)
    distances_img1 = np.abs(np.sum(lines_in_img1 * matched_points1, axis=1)) / np.sqrt(lines_in_img1[:, 0]**2 + lines_in_img1[:, 1]**2)

    # Compute mean distances
    mean_distance_img2 = np.mean(distances_img2)
    mean_distance_img1 = np.mean(distances_img1)

    print(f"Mean distance to epipolar lines in Image 2: {mean_distance_img2}")
    print(f"Mean distance to epipolar lines in Image 1: {mean_distance_img1}")

    # Determine F direction based on lower error
    if mean_distance_img2 < mean_distance_img1:
        print("F corresponds to F21 (points from Image 1 to lines in Image 2).")
        return "F21"
    else:
        print("F corresponds to F12 (points from Image 2 to lines in Image 1).")
        return "F12"

def compute_epipoles(F):
    """
        Compute the epipoles from the Fundamental Matrix.
        - Input:
            · F: Fundamental matrix (3x3)
        - Output:
            · Epipole in Image 1 and Epipole in Image 2
    """
    # Epipole in Image 1 (null space of F)
    _, _, Vt = np.linalg.svd(F)
    e1 = Vt[-1]  # Last row of V gives the right null space (epipole in Image 2)
    e1 /= e1[-1]  # Normalize to homogeneous coordinates

    # Epipole in Image 2 (null space of F^T)
    _, _, Vt = np.linalg.svd(F.T)
    e2 = Vt[-1]  # Last row of V' gives the right null space (epipole in Image 1)
    e2 /= e2[-1]  # Normalize to homogeneous coordinates

    return e1, e2

def compute_epipolar_distances(F, x1, x2):
    """
    Compute the epipolar distances between two sets of points given a fundamental matrix.
    - F: Fundamental matrix (3x3).
    - x1, x2: Points in image 1 and image 2, respectively (both are 3xN in homogeneous coordinates).
    - Returns: Array of epipolar distances for all points.
    """
    # Epipolar lines in Image 2 corresponding to points in Image 1
    l2 = (F @ x1).T  # Shape: (N, 3)
    
    # Ensure `l2` is of type float
    l2 = np.array(l2, dtype=np.float64)
    
    # Debugging: Check the shape of l2
    print("Shape of l2:", l2.shape)
    
    # Normalize the epipolar lines
    norms = np.linalg.norm(l2[:, :2], axis=1).reshape(-1, 1)
    
    # Debugging: Check the shape of norms
    print("Shape of norms:", norms.shape)
    
    # Ensure norms are not zero to avoid division by zero
    norms[norms == 0] = 1
    
    l2 /= norms
    
    # Compute distances for points in Image 2 to their corresponding epipolar lines
    distances = np.abs(np.sum(l2 * x2.T, axis=1))  # Shape: (N,)
    
    return distances

def median_epipolar_error(F, x1, x2):
    """
    Compute the median of the epipolar distances for a given fundamental matrix and point correspondences.
    - F: Fundamental matrix (3x3).
    - x1, x2: Points in image 1 and image 2, respectively (both are 3xN in homogeneous coordinates).
    - Returns: Median of the epipolar distances.
    """
    distances = compute_epipolar_distances(F, x1, x2)
    return np.median(distances)

def select_pose_with_lowest_epipolar_error(x1_h, x2_h, K1, K2, R1, R2, t):
    """
    Select the best pose by evaluating the median epipolar error.
    - x1_h, x2_h: Matched points in homogeneous coordinates (3xN).
    - K1, K2: Intrinsic matrices of Camera 1 and Camera 2.
    - R1, R2: Possible rotation matrices.
    - t: Translation vector.
    - Returns: The best rotation (R), translation (t), and epipolar error.
    """

    # Calculate F matrices for each pose
    F1 = np.linalg.inv(K2).T @ crossMatrix(t) @ R1 @ np.linalg.inv(K1)
    F2 = np.linalg.inv(K2).T @ crossMatrix(-t) @ R1 @ np.linalg.inv(K1)
    F3 = np.linalg.inv(K2).T @ crossMatrix(t) @ R2 @ np.linalg.inv(K1)
    F4 = np.linalg.inv(K2).T @ crossMatrix(-t) @ R2 @ np.linalg.inv(K1)

    # Array of Fs to return with idx
    Fs = [F1, F2, F3, F4]

    # Calculate the median epipolar error for each pose
    errors = [
        median_epipolar_error(F1, x1_h, x2_h),
        median_epipolar_error(F2, x1_h, x2_h),
        median_epipolar_error(F3, x1_h, x2_h),
        median_epipolar_error(F4, x1_h, x2_h)
    ]
    print(f"Median Epipolar Errors: {errors}")

    # Find the pose with the smallest median epipolar error
    best_idx = np.argmin(errors)
    poses = [(R1, t), (R1, -t), (R2, t), (R2, -t)]

    # if errors[best_idx] > 100:  # Adjust the threshold based on your data
    #     print("No valid pose found with low epipolar error!")
    #     return None, None, None, None

    return poses[best_idx][0], poses[best_idx][1], errors[best_idx], Fs[best_idx]

def own_PnP(pts_3D, pts_2D, K, initial_guess):
    """
    Simple PnP implementation using least squares optimization.
    :param pts_3D: Nx3 array of 3D points.
    :param pts_2D: Nx2 array of 2D image points.
    :param K: 3x3 camera intrinsic matrix.
    :return: Rotation matrix (3x3) and translation vector (3x1).
    """

    # Old initial guess for testing and debugging
    # # Initial guess: no rotation, translation based on the centroid of 3D points
    # centroid_3D = np.mean(pts_3D, axis=0)
    # initial_guess = np.zeros(6)
    # initial_guess[3:] = centroid_3D - (centroid_3D/2)  # Use the negative centroid as an initial guess for translation

    result = least_squares(reprojection_error_for_pnp, initial_guess, args=(pts_3D, pts_2D, K),
                           method="lm", verbose=2, ftol=1e-2)
    
    R_vec, t = result.x[:3], result.x[3:]
    return R_vec, t

def reprojection_error_for_pnp(params, pts_3D, pts_2D, K):
   
    R_vec, t = params[:3], params[3:]
    R = expm(crossMatrix(R_vec))
    T = ensamble_T(R, t)
    P = get_projection_matrix(K, T)
    
    pts_3D = pts_3D.T
    pts_3D_h = np.vstack((pts_3D, np.ones(pts_3D.shape[1])))
    projected_pts = project_to_camera(P, pts_3D_h)
    residuals = projected_pts[:2, :] - pts_2D[:]

    residuals= residuals.ravel()

    print(np.sum(np.abs(residuals)))

    return residuals


def triangulate_new_points_for_pair(db_name, adjacency, images_info,
                                    c1_id, c2_id,
                                    R_c2_c1, t_c2_c1,
                                    K, plot_residuals=False, img1=None, img2=None,
                                    T_c1_c2=None):

    # 1) Find untriangulated pairs
    new_matches = find_untriangulated_matches(images_info, adjacency, c1_id, c2_id)
    if not new_matches:
        print(f"No new matches to triangulate between {c1_id} and {c2_id}")
        return

    # 2) Gather pixel coords
    pts2d_c1 = []
    pts2d_c2 = []
    for (kp1, kp2) in new_matches:
        (c1, r1) = images_info[c1_id]["keypoints"][kp1]  # (col, row)
        (c2, r2) = images_info[c2_id]["keypoints"][kp2]
        # reorder to (x,y) => (col,row)
        pts2d_c1.append([c1, r1])
        pts2d_c2.append([c2, r2])

    pts2d_c1 = np.array(pts2d_c1, dtype=np.float64).T  # shape (2, N)
    pts2d_c2 = np.array(pts2d_c2, dtype=np.float64).T  # shape (2, N)

    # 3) Triangulate
    # P1 is identity matrix
    P1 = get_projection_matrix(K, np.eye(4))
    P2 = get_projection_matrix(K, ensamble_T(R_c2_c1, t_c2_c1))
    
    
    X_c1 = triangulate_points(pts2d_c1, pts2d_c2, P1, P2)
    print(f"Triangulated {X_c1.shape[1]} new 3D points from cameras {c1_id} & {c2_id}.")
    

    
    # TODO: compute resiudals and filter X , x1, x2 and match list
    X_c1, pts2d_c1, pts2d_c2, new_matches = compute_residulas_and_filter(pts2d_c1, pts2d_c2, X_c1, R_c2_c1, t_c2_c1, 
                                                                        new_matches, K, img1, img2, c1_id, c2_id,
                                                                          percentile_filter = 90, plot_FLAG = True)

    # TODO: filter X points with negative depth
    count, valid_depths1, valid_depths2 = count_valid_points(X_c1, P1, P2)
    X_c1_filtered = X_c1[:, valid_depths1 & valid_depths2]
    pts2d_c1 = pts2d_c1[:, valid_depths1 & valid_depths2]
    pts2d_c2 = pts2d_c2[:, valid_depths1 & valid_depths2]
    new_matches = [new_matches[i] for i in range(len(new_matches)) if valid_depths1[i] & valid_depths2[i]]

    print(f"Filtered out {X_c1.shape[1] - X_c1_filtered.shape[1]} points with negative depth.")

    # Filter outliers with a percentile of the norm of the X_c1_filtered 3D point
    # Calculate norm vector of X_c1_filtered
    norms = np.linalg.norm(X_c1_filtered[:3, :], axis=0)
    percentile_norm = np.percentile(norms, 90)
    filtered_indices = norms <= percentile_norm

    X_c1_norm_filtered = X_c1_filtered[:, filtered_indices]
    pts2d_c1 = pts2d_c1[:, filtered_indices]
    pts2d_c2 = pts2d_c2[:, filtered_indices]
    new_matches = [new_matches[i] for i in range(len(new_matches)) if filtered_indices[i]]

    print(f"Filtered out {X_c1_filtered.shape[1] - X_c1_norm_filtered.shape[1]} points with high norm.")
    print(f"Inserting {X_c1_norm_filtered.shape[1]} 3D points into the database.")



    if plot_residuals:
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(img1, pts2d_c1, project_to_camera(P1, X_c1_norm_filtered), f"Depth filtered Camera id {c1_id}", ax=axs[0], adjust_limits=False)
        visualize_residuals(img2, pts2d_c2, project_to_camera(P2, X_c1_norm_filtered),  f"Depth filteredCamera id {c2_id}", ax=axs[1], adjust_limits=False)
        plt.tight_layout()
        plt.show()

    # Change of coordinates
    if T_c1_c2 is not None:
        X_c1 = np.dot(T_c1_c2, X_c1)

    # 4) Insert 3D points into DB & memory
    insert_3d_points_in_memory_and_db(
        db_name, images_info,
        X_c1_norm_filtered, new_matches,
        c1_id, c2_id
    )

    print("Done inserting new 3D points.")

    return 


def triangulate_new_points_for_pair_in_c1(db_name, adjacency, images_info,
                                          c2_id, c3_id,
                                          R_c2_c1, t_c2_c1, R_c3_c1, t_c3_c1,
                                          K, plot_residuals=False, img2=None, img3=None):
    # 1) Find untriangulated pairs
    new_matches = find_untriangulated_matches(images_info, adjacency, c2_id, c3_id)
    if not new_matches:
        print(f"No new matches to triangulate between {c2_id} and {c3_id}")
        return

    # 2) Gather pixel coords
    pts2d_c2 = []
    pts2d_c3 = []
    for (kp2, kp3) in new_matches:
        (c2, r2) = images_info[c2_id]["keypoints"][kp2]
        (c3, r3) = images_info[c3_id]["keypoints"][kp3]
        pts2d_c2.append([c2, r2])
        pts2d_c3.append([c3, r3])

    pts2d_c2 = np.array(pts2d_c2, dtype=np.float64).T  # shape (2, N)
    pts2d_c3 = np.array(pts2d_c3, dtype=np.float64).T  # shape (2, N)

    # 3) Triangulate in C1 frame
    T_c2_c1 = ensamble_T(R_c2_c1, t_c2_c1)
    T_c3_c1 = ensamble_T(R_c3_c1, t_c3_c1)

    P_c2 = get_projection_matrix(K, T_c2_c1)
    P_c3 = get_projection_matrix(K, T_c3_c1)
    
    X_c1 = triangulate_points(pts2d_c2, pts2d_c3, P_c2, P_c3)
    print(f"Triangulated {X_c1.shape[1]} new 3D points from cameras {c2_id} & {c3_id}.")

    T_c1_c2 = np.linalg.inv(T_c2_c1)
    T_c1_c3 = np.linalg.inv(T_c3_c1)

    visualize_3D_3cameras(T_c1_c2,T_c1_c3,X_c1)


    X_c1, pts2d_c2, pts2d_c3, new_matches = compute_residulas_and_filter_in_c1(pts2d_c2, pts2d_c3, X_c1, R_c2_c1, t_c2_c1, 
                                R_c3_c1, t_c3_c1, new_matches, K, img2, img3, c2_id, c3_id, 
                                percentile_filter = 90, plot_FLAG = True)

    # 4) Filter points with negative depths
    count, valid_depths2, valid_depths3 = count_valid_points(X_c1, P_c2, P_c3)
    X_c1_filtered = X_c1[:, valid_depths2 & valid_depths3]
    pts2d_c2 = pts2d_c2[:, valid_depths2 & valid_depths3]
    pts2d_c3 = pts2d_c3[:, valid_depths2 & valid_depths3]
    new_matches = [new_matches[i] for i in range(len(new_matches)) if valid_depths2[i] & valid_depths3[i]]
    
    print(f"Filtered out {X_c1.shape[1] - X_c1_filtered.shape[1]} points with negative depth.")
    
    # Filter outliers with a percentile of the norm of the X_c1_filtered 3D point
    # Calculate norm vector of X_c1_filtered
    norms = np.linalg.norm(X_c1_filtered[:3, :], axis=0)
    percentile_norm = np.percentile(norms, 90)
    filtered_indices = norms <= percentile_norm

    X_c1_norm_filtered = X_c1_filtered[:, filtered_indices]
    pts2d_c2 = pts2d_c2[:, filtered_indices]
    pts2d_c3 = pts2d_c3[:, filtered_indices]
    new_matches = [new_matches[i] for i in range(len(new_matches)) if filtered_indices[i]]

    print(f"Filtered out {X_c1_filtered.shape[1] - X_c1_norm_filtered.shape[1]} points with high norm.")
    print(f"Inserting {X_c1_norm_filtered.shape[1]} 3D points into the database.")



    if plot_residuals:
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(img2, pts2d_c2, project_to_camera(P_c2, X_c1_norm_filtered), f"Depth filtered Camera id {c2_id}", ax=axs[0], adjust_limits=False)
        visualize_residuals(img3, pts2d_c3, project_to_camera(P_c3, X_c1_norm_filtered),  f"Depth filteredCamera id {c3_id}", ax=axs[1], adjust_limits=False)
        plt.tight_layout()
        plt.show()

    visualize_3D_3cameras(T_c1_c3,T_c1_c2,X_c1_filtered)

    # Run a BA to refine the 3D points
    params = X_c1_norm_filtered[:3, :].flatten()

    def reprojection_error_for_ba(params, pts2d_c2, pts2d_c3, P_c2, P_c3):
        X = params.reshape(3, -1)
        X_h = np.vstack((X, np.ones(X.shape[1])))
        x2_proj = project_to_camera(P_c2, X_h)
        x3_proj = project_to_camera(P_c3, X_h)
        residuals_c2 = pts2d_c2[:2, :] - x2_proj[:2, :]
        residuals_c3 = pts2d_c3[:2, :] - x3_proj[:2, :]
        residuals = np.hstack((residuals_c2.ravel(), residuals_c3.ravel()))
        print
        return residuals

    result = least_squares(reprojection_error_for_ba,
        params,
        args=(pts2d_c2, pts2d_c3, P_c2, P_c3),
        method="lm",
        verbose=2,
        ftol=1e-2,
    )

    X_c1_optimized = result.x.reshape(3, -1)

    visualize_3D_3cameras(T_c1_c3,T_c1_c2,X_c1_optimized)

    # 5) Insert into DB
    insert_3d_points_in_memory_and_db(
        db_name, images_info,
        X_c1_optimized, new_matches,
        c2_id, c3_id
    )
    print("Done inserting new 3D points.")
    return


def compute_residulas_and_filter_in_c1(pts2d_c2, pts2d_c3, X_c1, R_c2_c1, t_c2_c1, 
                                R_c3_c1, t_c3_c1, match_list, K, img2, img3, c2_id, c3_id, 
                                percentile_filter = 90, plot_FLAG = True):
        
        # Visualize residuals between 2D points and reprojection of 3D points
        T_c2_c1 = ensamble_T(R_c2_c1,t_c2_c1)
        P_c2_c1 = get_projection_matrix(K, T_c2_c1)
        x2_proj = project_to_camera(P_c2_c1, X_c1)

        T_c3_c1 = ensamble_T(R_c3_c1,t_c3_c1)
        P_c3_c1 = get_projection_matrix(K,T_c3_c1)
        x3_proj = project_to_camera(P_c3_c1, X_c1)

        if plot_FLAG:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            visualize_residuals(img2, pts2d_c2, x2_proj, f"Initial Residuals in Image id {c2_id}", ax=axs[0], adjust_limits= False)
            visualize_residuals(img3, pts2d_c3, x3_proj, f'Initial Residuals in Image id {c3_id}', ax=axs[1], adjust_limits= False)
            plt.tight_layout()
            plt.show()

        res_x2 = pts2d_c2[:2,:] - x2_proj[:2,:]
        res_x3 = pts2d_c3[:2,:] - x3_proj[:2,:]

        # Compute the norm of each vector in res_x1
        res_x2_norms = np.linalg.norm(res_x2, axis=0)
        res_x3_norms = np.linalg.norm(res_x3, axis=0)

        # Calculate the 95th percentile of the norms
        percentile_res_x2 = np.percentile(res_x2_norms, percentile_filter)
        percentile_res_x3 = np.percentile(res_x3_norms, percentile_filter)

        # Filter the residuals based on the 95th percentile
        filtered_indices_res_x2 = res_x2_norms <= percentile_res_x2
        filtered_indices_res_x3 = res_x3_norms <= percentile_res_x3

        #Filter all the points
        X_c1_filtered = X_c1[:, filtered_indices_res_x2 & filtered_indices_res_x3]
        pts2d_c2 = pts2d_c2[:, filtered_indices_res_x2 & filtered_indices_res_x3]
        pts2d_c3 = pts2d_c3[:, filtered_indices_res_x2 & filtered_indices_res_x3]
        x2_proj = x2_proj[:, filtered_indices_res_x2 & filtered_indices_res_x3]
        x3_proj = x3_proj[:, filtered_indices_res_x2 & filtered_indices_res_x3]
        match_list = [match_list[i] for i in range(len(match_list)) if filtered_indices_res_x2[i] & filtered_indices_res_x3[i]]

        #Print percentage of points filtered
        print(f"{X_c1.shape[1] - X_c1_filtered.shape[1]} points with high initial residuals filtered out of {X_c1.shape[1]}")


        if plot_FLAG:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            visualize_residuals(img2, pts2d_c2, x2_proj, f"Filtered Residuals in Image id {c2_id}", ax=axs[0], adjust_limits= False)
            visualize_residuals(img3, pts2d_c3, x3_proj, f'Filtered Residuals in Image id {c3_id}', ax=axs[1], adjust_limits= False)
            plt.tight_layout()
            plt.show()

        return X_c1_filtered, pts2d_c2, pts2d_c3, match_list
    


def compute_residulas_and_filter(x1_h, x2_h, X_c1_initial, R_c2_c1_initial, t_c2_c1_initial, match_list, K, 
                                img1, img2, c_id_1, c_id_2, percentile_filter = 90, plot_FLAG = True):

        # Visualize residuals between 2D points and reprojection of 3D points
        #From World to image4 that is c1
        P_c1_c1_initial = get_projection_matrix(K, np.eye(4))
        x1_proj_initial = project_to_camera(P_c1_c1_initial, X_c1_initial)
        # Convert X1 to homogeneous coordinates (4 x nPoints)

        #From World to image2
        T_c2_c1_initial = ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
        P_c2_c1_initial = get_projection_matrix(K,T_c2_c1_initial)
        x2_proj_initial = project_to_camera(P_c2_c1_initial, X_c1_initial)


        if plot_FLAG:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            visualize_residuals(img1, x1_h, x1_proj_initial, f"Initial Residuals in Image id {c_id_1}", ax=axs[0], adjust_limits= False)
            visualize_residuals(img2, x2_h, x2_proj_initial, f'Initial Residuals in Image id {c_id_2}', ax=axs[1], adjust_limits= False)
            plt.tight_layout()
            plt.show()

        # Compute residuals
        res_x1 = x1_h[:2,:] - x1_proj_initial[:2]
        res_x2 = x2_h[:2,:] - x2_proj_initial[:2]

        # Filter out the points in X_c1_initial with high residuals
        # Compute the norm of each vector in res_x1
        res_x1_norms = np.linalg.norm(res_x1, axis=0)
        res_x2_norms = np.linalg.norm(res_x2, axis=0)

        # Calculate the 95th percentile of the norms
        percentile_res_x1 = np.percentile(res_x1_norms, percentile_filter)
        percentile_res_x2 = np.percentile(res_x2_norms, percentile_filter)

        # Filter the residuals based on the 95th percentile
        filtered_indices_res_x1 = res_x1_norms <= percentile_res_x1
        filtered_indices_res_x2 = res_x2_norms <= percentile_res_x2

        #Filter all the points
        X_c1 = X_c1_initial[:, filtered_indices_res_x1 & filtered_indices_res_x2]
        x1_h = x1_h[:, filtered_indices_res_x1 & filtered_indices_res_x2]
        x2_h = x2_h[:, filtered_indices_res_x1 & filtered_indices_res_x2]
        x1_proj_initial = x1_proj_initial[:, filtered_indices_res_x1 & filtered_indices_res_x2]
        x2_proj_initial = x2_proj_initial[:, filtered_indices_res_x1 & filtered_indices_res_x2]
        match_list = [match_list[i] for i in range(len(match_list)) if filtered_indices_res_x1[i] & filtered_indices_res_x2[i]]

        #Print percentage of points filtered
        print(f"{X_c1_initial.shape[1] - X_c1.shape[1]} points with high initial residuals filtered out of {X_c1_initial.shape[1]}")


        if plot_FLAG:
            fig, axs = plt.subplots(1, 2, figsize=(18, 6))
            visualize_residuals(img1, x1_h, x1_proj_initial, f"Filtered Residuals in Image id {c_id_1}", ax=axs[0], adjust_limits= False)
            visualize_residuals(img2, x2_h, x2_proj_initial, f'Filtered Residuals in Image id {c_id_2}', ax=axs[1], adjust_limits= False)
            plt.tight_layout()
            plt.show()

        return X_c1, x1_h, x2_h, match_list

def get_points_seen_by_camera(database_path, images_info, c_id_3,c_id_1, match_list_3_1, pnp_points_2d, pnp_points_3d, x1_for_pnp):

    for (kp3, kp1) in match_list_3_1:
        if kp1 in images_info[c_id_1]["kp3D"]: 
            p3d_id = images_info[c_id_1]["kp3D"][kp1]
            if p3d_id is not None:
                # Query its 3D coordinates from DB or store it in memory
                X, Y, Z = get_3d_point_coordinates(database_path, p3d_id)
                (c3, r3) = images_info[c_id_3]["keypoints"][kp3] 
                (c1, r1) = images_info[c_id_1]["keypoints"][kp1]
                pnp_points_2d.append([c3, r3])
                pnp_points_3d.append([X, Y, Z])
                x1_for_pnp.append([c1, r1])
    
    return pnp_points_2d, pnp_points_3d, x1_for_pnp


#endregion

    
#################################################### PLOTTING FUNCTIONS ####################################################
#region Plotting FUNCTIOS
# BUG: Check if the plotting is correct: should use the inverse T for rt
def draw_possible_poses(ax, T_w_c, X):
    fig3D = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_w_c, '-', 'C2')
    ax.scatter(X[0, :], X[1, :], X[2, :], marker='.', color='r')

    return fig3D

def adjust_plot_limits(ax, X):
    """
    Adjust the plot limits based on the range of 3D points.
    - Input:
        · ax: The 3D plot axis.
        · X: Triangulated 3D points (non-homogeneous coordinates).
    """
    X_cartesian = X[:3, :] / X[3, :]  # Convert to non-homogeneous coordinates
    x_min, x_max = np.min(X_cartesian[0, :]), np.max(X_cartesian[0, :])
    y_min, y_max = np.min(X_cartesian[1, :]), np.max(X_cartesian[1, :])
    z_min, z_max = np.min(X_cartesian[2, :]), np.max(X_cartesian[2, :])

    # Set plot limits
    ax.set_xlim([x_min - 10, x_max + 10])
    ax.set_ylim([y_min - 10, y_max + 10])
    ax.set_zlim([z_min - 10, z_max + 10])

def plotResidual(x,xProjected,strStyle):
    """
        Plot the residual between an image point and an estimation based on a projection model.
         -input:
             x: Image points.
             xProjected: Projected points.
             strStyle: Line style.
         -output: None
         """

    for k in range(x.shape[1]):
        plt.plot([x[0, k], xProjected[0, k]], [x[1, k], xProjected[1, k]], strStyle)

def plotNumberedImagePoints(x,strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset, x[1, k]+offset, str(k), color=strColor)

def plotNumbered3DPoints(ax, X,strColor, offset):
    """
        Plot indexes of points on a 3D plot.
         -input:
             ax: axis handle
             X: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(X.shape[1]):
        ax.text(X[0, k]+offset, X[1, k]+offset, X[2,k]+offset, str(k), color=strColor)

def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
    Draw a segment in a 3D plot
    -input:
        ax: axis handle
        xIni: Initial 3D point.
        xEnd: Final 3D point.
        strStyle: Line style.
        lColor: Line color.
        lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
    -input:
        ax: axis handle
        T_w_c: (4x4 matrix) Reference system C seen from W.
        strStyle: lines style.
        nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)           

def visualize_residuals(image, observed_points, projected_points, title, ax=None, adjust_limits=True):
    """
    Visualize residuals between observed and projected points on an image.
    
    Parameters:
        image (np.array): The image on which to plot the points.
        observed_points (np.array): The observed 2D points.
        projected_points (np.array): The projected 2D points.
        title (str): The title of the plot.
        ax (matplotlib.axes.Axes, optional): The axes on which to plot. If None, create a new figure and axes.
    """
    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(image, cmap='gray')
    ax.scatter(observed_points[0, :], observed_points[1, :], color='r', marker='x', label='Observed Points')
    ax.scatter(projected_points[0, :], projected_points[1, :], color='b', label='Projected Points')
    for i in range(observed_points.shape[1]):
        ax.plot([observed_points[0, i], projected_points[0, i]], [observed_points[1, i], projected_points[1, i]], 'g-')
    
    if adjust_limits:
        ax.set_xlim([0, image.shape[1]])
        ax.set_ylim([image.shape[0], 0])
    
    ax.set_title(title)
    ax.legend()

def visualize_3D_3cameras(T_c1_c2, T_c1_c3, X, adjust_plot_limits=True):
    """
    Plot the 3D cameras and the 3D points.
    """
    plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_c1_c2, '-', 'C2')
    drawRefSystem(ax, T_c1_c3, '-', 'C3')

    ax.scatter(X[0, :], X[1, :], X[2, :], marker='.')
    # plotNumbered3DPoints(ax, X_w, 'r', 0.1)

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    
    # Optional: set plot limits to manage scale and view
    if adjust_plot_limits:
        ax.set_xlim([2,-2])
        ax.set_ylim([2,-2])
        ax.set_zlim([3,-1])


    print('Close the figure to continue. Left button for orbit, right button for zoom.')
    plt.show()  

def visualize_3D_c1_2cameras(T_c2_c1, X_c1_w):
    """
    Plot the 3D cameras and the 3D points.
    """
    plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, T_c2_c1, '-', 'C2')

    ax.scatter(X_c1_w[0, :], X_c1_w[1, :], X_c1_w[2, :], marker='.')
    # plotNumbered3DPoints(ax, X_c1_w, 'r', 0.1)

    # Matplotlib does not correctly manage the axis('equal')
    xFakeBoundingBox = np.linspace(0, 4, 2)
    yFakeBoundingBox = np.linspace(0, 4, 2)
    zFakeBoundingBox = np.linspace(0, 4, 2)
    plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
    
    # Optional: set plot limits to manage scale and view
    ax.set_xlim([2,-2])
    ax.set_ylim([2,-2])
    ax.set_zlim([3,-1])


    print('Close the figure to continue. Left button for orbit, right button for zoom.')

    plt.show()  

def visualize_3D_comparison(ax, T_c1_c2_initial, X_c1_w_initial, T_c1_c2_opt, X_c1_w_opt):
    """
    Plot the 3D cameras and the 3D points for ground truth, initial guess, and optimized solution.
    - Ground truth: uses T_w_c1, T_w_c2, X_w
    - Initial: uses T_c1_c2_initial, X_c1_w_initial
    - Optimized: uses T_c1_c2_opt, X_c1_w_opt
    """

    # Initial Guess (Camera 1 Frame)
    drawRefSystem(ax, np.eye(4), '-', 'C1')
    drawRefSystem(ax, T_c1_c2_initial, '--', 'C2 (Initial)')
    ax.scatter(X_c1_w_initial[0, :], X_c1_w_initial[1, :], X_c1_w_initial[2, :], marker='^', color='b', label='3D Points (Initial)')

    # Optimized Solution (Camera 1 Frame)
    drawRefSystem(ax, T_c1_c2_opt, '-.', 'C2 (Optimized)')
    ax.scatter(X_c1_w_opt[0, :], X_c1_w_opt[1, :], X_c1_w_opt[2, :], marker='x', color='r', label='3D Points (Optimized)')

    # Labels and Legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()

    # Optional: set plot limits to manage scale and view
    ax.set_xlim([2,-2])
    ax.set_ylim([2,-2])
    ax.set_zlim([3,-1])

    plt.show()

def visualize_3D_3cameras_optimization(T_w_c1, T_w_c2, T_w_c3, X_w, T_c1_c2_initial, T_c1_c3_initial, T_c1_c2_opt, T_c1_c3_opt, X_c1_w_initial, X_c1_w_opt):
    """
    Visualize 3D results for ground truth, initial guess, and optimized solution.

    - T_w_c1, T_w_c2, T_w_c3: Ground truth transformation matrices for cameras 1, 2, and 3.
    - X_w: Ground truth 3D points in the world coordinate system.
    - T_c2_c1_initial, T_c3_c1_initial: Initial transformation matrices for cameras 2 and 3 with respect to camera 1.
    - T_c2_c1_opt, T_c3_c1_opt: Optimized transformation matrices for cameras 2 and 3 with respect to camera 1.
    - X_initial: Initial 3D points.
    - X_opt_scaled: Optimized 3D points, scaled.
    """

    # Transform ground truth points and cameras into Camera 1's frame
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c1_c2_gt = T_c1_w @ T_w_c2  # Ground truth transformation of Camera 2 in Camera 1's frame
    T_c1_c3_gt = T_c1_w @ T_w_c3  # Ground truth transformation of Camera 3 in Camera 1's frame
    X_c1_w_gt = T_c1_w @ X_w        # Transform ground truth points into Camera 1's frame
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Ground truth points and cameras
    drawRefSystem(ax, np.eye(4), '-', 'C1 (Ground Truth)')
    drawRefSystem(ax, T_c1_c2_gt, '-', 'C2 (Ground Truth)')
    drawRefSystem(ax, T_c1_c3_gt, '-', 'C3 (Ground Truth)')
    ax.scatter(X_c1_w_gt[0, :], X_c1_w_gt[1, :], X_c1_w_gt[2, :], marker='o', color='g', label='3D Points (Ground Truth)')

    # Initial guess points and cameras
    drawRefSystem(ax, T_c1_c2_initial, '--', 'C2 (Initial)')
    drawRefSystem(ax, T_c1_c3_initial, '--', 'C3 (Initial)')
    ax.scatter(X_c1_w_initial[0, :], X_c1_w_initial[1, :], X_c1_w_initial[2, :], marker='^', color='b', label='3D Points (Initial)')

    # Optimized points and cameras
    drawRefSystem(ax, T_c1_c2_opt, '-.', 'C2 (Optimized)')
    drawRefSystem(ax, T_c1_c3_opt, '-.', 'C3 (Optimized)')
    ax.scatter(X_c1_w_opt[0, :], X_c1_w_opt[1, :], X_c1_w_opt[2, :], marker='x', color='r', label='3D Points (Optimized)')

    # Set plot labels and legend
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.legend()
    plt.title("3D Visualization of Ground Truth, Initial, and Optimized Results")

    # Optional: set plot limits to manage scale and view
    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()

def plot_epipolar_line(F, x1, ax2, img2, show_epipoles):
    """
        Given a fundamental matrix and a point in image 1, plot the corresponding epipolar line in image 2.
        Also, plot the epipole in image 2 if show_epipoles is True.
        - Input:
            · F (np.array): Fundamental matrix (3x3)
            · x1 (np.array): Point in image 1 (homogeneous coordinates, shape (3,))
            · ax2: Axis for image 2 where the epipolar line will be plotted
            · img2: Image 2 (for replotting as background)
    """
    # Compute the epipolar line in image 2
    l2 = F @ x1  # l2 = [a, b, c], where the line equation is ax + by + c = 0

    if show_epipoles:
        # Create a range of x values for image 2 chosing the max between image width and the epipole x coordinate
        e1, e2 = compute_epipoles(F)
        height, width = img2.shape[:2]
        x_vals = np.array([0, max(width, int(e2[0]))])

        ax2.plot(e2[0], e2[1], 'rx', markersize=10)  # Red "x" at the epipole in image 2
        ax2.text(e2[0], e2[1], 'epipole', color='r', fontsize=12)

    else: 
         # Create a range of x values for image 2
        height, width = img2.shape[:2]
        x_vals = np.array([0, width])
    
    # Calculate the corresponding y values for the epipolar line
    y_vals = -(l2[0] * x_vals + l2[2]) / l2[1]
    
    ax2.plot(x_vals, y_vals, 'r')  # Plot the epipolar line
    ax2.set_title('Image 2 - Epipolar Lines')
    plt.draw()  # Redraw the figure to update the plot

def on_click(event, F, ax1, ax2, img2, show_epipoles):
    """
        Event handler for mouse click. Computes and plots the epipolar line in image 2 based on the click in image 1.
        - Input
            · event: The mouse event (contains the click coordinates)
            · F: Fundamental matrix
            · ax1, ax2: Axis for images 1 and 2 where the epipolar line will be plotted
            · img2: Image 2 (for replotting as background)
    """
    # Get the click coordinates
    x_clicked = event.xdata
    y_clicked = event.ydata
    
    # Mark the clicked point in image 1 with an "x"
    ax1.plot(x_clicked, y_clicked, 'rx', markersize=10)  # Red "x" at the clicked point
    ax1.set_title('Image 1 - Select Point')
    
    if x_clicked is not None and y_clicked is not None:
        print(f"Clicked coordinates in Image 1: ({x_clicked:.2f}, {y_clicked:.2f})")
        
        # Convert the clicked point to homogeneous coordinates
        x1_clicked = np.array([x_clicked, y_clicked, 1])
        
        # Plot the corresponding epipolar line in image 2
        plot_epipolar_line(F, x1_clicked, ax2, img2, show_epipoles)

def visualize_epipolar_lines(F, img1, img2, show_epipoles=False):
    """
        Visualize epipolar lines in two images given a fundamental matrix F.
        Clicking on image 1 will plot the corresponding epipolar line in image 2.
        - Input:
            · F (np.array): Fundamental matrix (3x3)
            · img1, img2: Images 1 and 2
            · show_epipoles (bool): Whether to plot the epipoles in both images.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))
    # Configuración del primer subplot
    ax1.set_xlabel('Coordinates X (píxeles)')
    ax1.set_ylabel('Coordinates Y (píxeles)')
    ax1.imshow(img1)
    ax1.set_xlim([0, img1.shape[1]])
    ax1.set_ylim([img1.shape[0], 0])
    ax1.set_title('Image 1 - Select Point')
    
    # Segundo subplot para la segunda imagen
    ax2.set_xlabel('Coordinates X (píxeles)')
    ax2.set_ylabel('Coordinates Y (píxeles)')
    ax2.imshow(img2)
    ax2.set_xlim([0, img2.shape[1]])
    ax2.set_ylim([img2.shape[0], 0])
    ax2.set_title('Image 2 - Epipolar Lines')

    # Connect the click event on image 1 to the handler
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, F , ax1, ax2, img2,show_epipoles))
    print('\nClose the figure to continue. Select a point from Img1 to get the equivalent epipolar line.')
    plt.show()

# Visualize  residuals
def visualize_residuals_from_cameras(obs_list, points_3d_dict, camera_data, K, images_list, c_ids):
    fig, axs = plt.subplots(1, len(camera_data), figsize=(18, 6))
    for cid, cinfo in camera_data.items():
        rvec = cinfo["rvec"]
        tvec = cinfo["tvec"]
        R = expm(crossMatrix(rvec))
        T = ensamble_T(R, tvec)
        P = get_projection_matrix(K, T)
        img_points = []
        proj_points = []
        for obs in obs_list:
            if obs[0] == cid:
                pid = obs[1]
                x_meas = obs[2]
                y_meas = obs[3]
                X = points_3d_dict[pid]
                X_h = np.hstack((X, 1))
                X_h = np.array([X_h])
                X_h = X_h.T
                x_proj = project_to_camera(P, X_h)
                x_proj = x_proj[:2]
                x_proj = x_proj.reshape(-1)
                x_proj = x_proj.tolist()
                img_points.append([x_meas, y_meas])
                proj_points.append(x_proj)
        img_points = np.array(img_points).T
        proj_points = np.array(proj_points).T
        visualize_residuals(images_list[c_ids.index(cid)], img_points, proj_points, f"Initial Residuals in Image {cid}", ax=axs[c_ids.index(cid)], adjust_limits=False)
    plt.tight_layout()
    plt.show()

#endregion

   
#################################################### BUNDLE FUNCTIONS ####################################################
#region BUNDLE FUNCTIONS

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints, bundle_method="lm"):
    """
    -input:
    Op: Optimization parameters: this must include a paramtrization for T_21 
    in a proper way and for X1 (3D points in ref 1)
    x1Data: (3xnPoints) 2D points on image 1 
    x2Data: (3xnPoints) 2D points on image 2 
    K_c: (3x3) Intrinsic calibration matrix 
    nPoints: Number of points
    -output:
    res: residuals from the error between the 2D matched points  and the projected points from the 3D points
    (2 equations/residuals per 2D point)
    """

# Extract rotation vector (theta) and translation vector (t) from Op
# TODO: Habria que ajustar escala con vector unitario de t
 
    theta = Op[:3]              # Rotation vector (Rvec)
    # t = Op[3:6]
    # Translation vector
    t_theta = Op[3]
    t_phi = Op [4]
    t = np.array([np.sin(t_theta)*np.cos(t_phi), np.sin(t_theta)*np.sin(t_phi), np.cos(t_theta)])   # Rodrigues vector


    X = Op[5:].reshape(3,-1)  # Reshape 3D points to (nPoints, 3)
    
    # Convert rotation vector to rotation matrix using matrix exponential
    R = expm(crossMatrix(theta))

    # Define the projection matrix for camera 1 (identity rotation and zero translation)
    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))

    T2 = ensamble_T(R, t)
    P2 = get_projection_matrix(K_c, T2)
    
    
    # Convert X1 to homogeneous coordinates (4 x nPoints)
    # X_h = np.vstack((X.T, np.ones((1, nPoints))))
    X_h = np.vstack((X, np.ones((1, nPoints))))
    
    # Project 3D points to camera 1 and 2
    x1_projected = project_to_camera(P1, X_h)
    x2_projected = project_to_camera(P2, X_h)

    # Compute residuals
    res_x1 = x1Data[:2,:] - x1_projected[:2]
    res_x2 = x2Data[:2,:] - x2_projected[:2]

    res_x1_total = np.sum(np.abs(res_x1))
    res_x2_total = np.sum(np.abs(res_x1))

    if bundle_method == "lm":
        print("\n residuals 1:")
        print(res_x1_total)
        print("\n residuals 2:")
        print(res_x2_total)

    residuals = np.hstack((res_x1.flatten(), res_x2.flatten()))


    return residuals

def resBundleProjection3Views12DoF(Op, x1Data, x2Data, x3Data, K_c, nPoints):
    """
    Residual function for bundle adjustment with a chained transformation,
    where Camera 3 is defined relative to Camera 2.
    
    Parameters:
        Op (np.array): Optimization parameters [theta_21, t_21, theta_32, t_32, X (3D points)].
        x1Data, x2Data, x3Data (np.array): Observed 2D points for Cameras 1, 2, and 3.
        K_c (np.array): Intrinsic calibration matrix.
        nPoints (int): Number of points.
    
    Returns:
        np.array: Residuals for bundle adjustment.
    """


    # Extract parameters
    theta_21 = Op[:3]               # Rotation from Camera 1 to Camera 2
    t_21 = Op[3:6]                  # Translation from Camera 1 to Camera 2
    theta_31 = Op[6:9]              # Rotation from Camera 1 to Camera 3
    t_31 = Op[9:12]                 # Translation from Camera 1 to Camera 3
    X_c1_w = Op[12:].reshape(3,-1)      # 3D points in the reference frame of Camera 1

    # Convert rotation vectors to rotation matrices
    R_21 = expm(crossMatrix(theta_21))
    R_31 = expm(crossMatrix(theta_31))
    T_21 = ensamble_T(R_21, t_21)
    T_31 = ensamble_T(R_31, t_31)
    
    # Define projection matrix for Camera 1 (identity rotation and zero translation)
    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))
    P2 = get_projection_matrix(K_c, T_21)
    P3 = get_projection_matrix(K_c, T_31)


    # Convert X to homogeneous coordinates (4 x nPoints)
    X_c1_w_h = np.vstack((X_c1_w, np.ones((1, nPoints))))
    
    # Project 3D points to Camera 1, 2, and 3
    x1_projected = project_to_camera(P1, X_c1_w_h) 
    x2_projected = project_to_camera(P2, X_c1_w_h)
    x3_projected = project_to_camera(P3, X_c1_w_h)

    # Compute residuals for each camera
    res_x1 = x1Data[:2, :] - x1_projected[:2, :]
    res_x2 = x2Data[:2, :] - x2_projected[:2, :]
    res_x3 = x3Data[:2, :] - x3_projected[:2, :]

    res_x1_total = np.sum(np.abs(res_x1))
    res_x2_total = np.sum(np.abs(res_x2))
    res_x3_total = np.sum(np.abs(res_x3))

    # Stack all residuals into a single vector
    residuals = np.hstack((res_x1.flatten(), res_x2.flatten(), res_x3.flatten()))

    print("\n residuals 1:")
    print(res_x1_total)
    print("\n residuals 2:")
    print(res_x2_total)
    print("\n residuals 3:")
    print(res_x3_total)

    return residuals

def run_incremental_ba(camera_data, obs_list, points_3d_dict, K):
    """
    camera_data: a dict containing 'fixed' or 'free' for each camera, 
                 plus rvec/tvec initial guess, plus 'index' for free cameras
    obs_list: all observations (image_id, point3D_id, x_meas, y_meas)
    points_3d_dict: point3D_id -> (X, Y, Z)
    K: (3,3) intrinsics

    returns optimized param vector
    """
    # 1) build initial param vector
    params_init, unique_pids = build_parameter_vector(camera_data, points_3d_dict)

    # 2) build a map of p3d_id -> index in points array
    points_3d_index = {pid: i for i, pid in enumerate(unique_pids)}


    # # 3) define residual func
    def residual_func(p):

        return  residual_function_generic(p, obs_list, points_3d_index, camera_data, K)
    

    
    # 4) run BA
    result = least_squares(residual_func,
                            params_init, 
                            method='lm',
                            verbose=2,
                            ftol = 1e-1,
                                                    )
    
    np.save("result", result.x)
    
    print("BA finished:", result.success, result.message)

    return result.x

def residual_function_generic(params, obs_list, points_3d_index, camera_data, K):
    """
    - params: array containing 6*N_free camera params + 3*N_points for 3D coords
    - obs_list: list of (image_id, point3D_id, x_meas, y_meas)
    - points_3d_index: { point3D_id -> index_in_sorted_list }
    - camera_data: dictionary that tells us if a camera is fixed or free, plus index
    - K: intrinsics
    returns: residual vector of shape (2 * len(obs_list),)
    """
    # 1) how many cameras are free?
    free_cam_ids = [cid for cid, info in camera_data.items() if not info["fixed"]]
    n_free = len(free_cam_ids)
    offset_3d = 6*n_free  # after all camera blocks

    residuals = []

    for (img_id, p3d_id, x_meas, y_meas) in obs_list:
        # get camera R, t
        R, t = get_camera_block(params, camera_data, img_id)

        # retrieve the 3D coords from the param vector 
        i3d = points_3d_index[p3d_id]
        start_3d = offset_3d + 3*i3d
        X_3D = params[start_3d : start_3d+3]

        # build a 4D point
        # X_h = np.array([X_3D[0], X_3D[1], X_3D[2], 1.0], dtype=np.float64)
        X_h = np.array([X_3D[0], X_3D[1], X_3D[2], 1.0], dtype=np.float64).reshape(-1, 1)

        # form P = K [R | t]
        T = ensamble_T(R, t)
        P = get_projection_matrix(K, T)
        x_h_proj = project_to_camera(P, X_h)

        x_res = np.abs(x_meas -  x_h_proj[0])
        y_res = np.abs(y_meas - x_h_proj[1])
        res_total = x_res + y_res
        residuals.append(x_res)
        residuals.append(y_res)
        # print(f"Residuals for image {img_id} and point {p3d_id}: {res_total}")

    # print("Residuals sum:")
    print(np.sum(np.abs(residuals)))
    residuals = np.array(residuals).flatten()
    return residuals

def get_camera_block(params, camera_data, camera_id):
    """
    Return (R, t) for the specified camera.
    If the camera is fixed, we read from camera_data.
    If free, we slice from the params vector.
    """
    # If it's fixed, just parse camera_data['rvec','tvec'] and convert to R
    if camera_data[camera_id]["fixed"]:
        rvec = camera_data[camera_id]["rvec"]  
        tvec = camera_data[camera_id]["tvec"]
        #if rvec = np.zeros, R is identity
        if np.all(rvec == 0):
            R = np.eye(3)
        else:
            R = expm(crossMatrix(rvec))

        return R, tvec

    # Otherwise, it's free. We look up the block index:
    idx = camera_data[camera_id]["index"]
    start = idx*6
    rvec = params[start:start+3]
    tvec = params[start+3:start+6]
    R = expm(crossMatrix(rvec))

    return R, tvec

def build_parameter_vector(camera_data, points_3d_dict):
    """
    Build a parameter vector from the camera_data dictionary and the 3D points.
    We only store 'free' cameras (fixed cameras are not included).
    """
    # 1) Figure out how many cameras are free
    free_cam_ids = [cid for cid, caminfo in camera_data.items() if not caminfo["fixed"]]
    # Sort them by 'index' to have a consistent ordering
    free_cam_ids.sort(key=lambda c: camera_data[c]["index"])

    # 2) Build camera param array
    cam_params_list = []
    for cid in free_cam_ids:
        # each free camera has rvec, tvec
        rvec_init = camera_data[cid]["rvec"]
        tvec_init = camera_data[cid]["tvec"]
        cam_params_list.append(rvec_init)
        cam_params_list.append(tvec_init)
    camera_params_init = np.concatenate(cam_params_list, axis=0)  # shape (6*N_free,)

    # 3) Build points array
    unique_pids = sorted(points_3d_dict.keys())
    points_3d_list = []
    for pid in unique_pids:
        X, Y, Z = points_3d_dict[pid]
        points_3d_list.append([X, Y, Z])
    points_3d_init = np.array(points_3d_list).flatten()  # shape (3*N_points,)

    # Combine
    params_init = np.concatenate([camera_params_init, points_3d_init], axis=0)
    return params_init, unique_pids

def chunked_least_squares(fun, x0, chunk_evals=50, max_chunks=10, **kwargs):
    x_current = np.copy(x0)
    for chunk_idx in range(max_chunks):
        try:
            res = least_squares(fun, x_current,
                                max_nfev=chunk_evals,
                                **kwargs)
        except KeyboardInterrupt:
            # If user hits Ctrl+C mid-chunk, we don't get a partial solution.
            print("Interrupted mid-chunk. No new partial result available!")
            break

        # Get the partial result after finishing this chunk of max_nfev
        x_current = res.x  
        cost = 0.5 * np.sum(res.fun**2)
        print(f"Chunk {chunk_idx+1}: cost={cost}, status={res.status}, message='{res.message}'")
        
        # Save partial result to disk
        np.save(f"ba_params_chunk_{chunk_idx+1}.npy", x_current)
        
        if res.status > 0:
            # It converged or otherwise stopped early => done
            break

    return x_current



#endregion


#################################################### SQLITE/ CORRESPONDENCE FUNCTIONS ####################################################
#region SQLITE FUNCTIONS

def extract_R_t_from_F(db_name, image_id1, image_id2, K):

    """
    Extract R and t from the F matrix stored in the database for a pair of images.
    :param db_name: Path to the database.
    :param image_name1: First image name.
    :param image_name2: Second image name.
    :param K: Camera intrinsic matrix.
    :return: Possible rotation matrices (R1, R2) and translation vector (t).
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    inverse_id_FLAG = False

    # Retrieve the F matrix for the image pair
    pair_id = f"{image_id1}_{image_id2}"
    cursor.execute("SELECT F FROM two_view_geometries WHERE pair_id = ?;", (pair_id,))
    result = cursor.fetchone()

    if result is None:
        # Try with the names interchanged
        pair_id = f"{image_id2}_{image_id1}"
        cursor.execute("SELECT F FROM two_view_geometries WHERE pair_id = ?;", (pair_id,))
        result = cursor.fetchone()
        print("Inverse pair id found")
        inverse_id_FLAG = True


    if result is None:
        conn.close()
        raise ValueError(f"Fundamental matrix not found for pair: {image_id1} and {image_id2}")

    
    F_json = result[0]
    F = np.array(json.loads(F_json))
   
    if not inverse_id_FLAG:
        F = F.T

    # Compute the Essential matrix
    E = compute_essential_matrix_from_F(F, K, K)

    # Decompose the Essential matrix to extract R and t
    R1, R2, t = decompose_essential_matrix(E)

    conn.close()
    return R1, R2, t, F

def get_camera_intrinsics(db_name):
    """
    Retrieve camera parameters from the database and construct the K matrix for a pinhole model.
    
    :param db_name: Path to the SQLite database.
    :return: Camera intrinsic matrix (K) as a numpy array.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    
    # Query the camera parameters (assuming only one camera is used)
    cursor.execute("SELECT params FROM Cameras LIMIT 1;")
    result = cursor.fetchone()
    
    if result is None:
        conn.close()
        raise ValueError("No camera parameters found in the database.")
    
    # Parse the intrinsic parameters
    params = result[0]
    fx, fy, cx, cy = map(float, params.split(", "))
    
    # Construct the intrinsic matrix for the pinhole camera model
    K = np.array([
        [fx, 0, cx],
        [0, fy, cy],
        [0, 0, 1]
    ])
    
    conn.close()
    return K

#BUG: retrieving second match incorrectly. FIXED: use adjancey graph from in-memory structure
def retrieve_matched_points_with_pair_id(db_name, image_name1, image_name2):
    """
    Retrieve matched 2D points (x1, x2) from the database using `pair_id`.
    This version includes the following steps:
    1. Retrieves `pair_id` from `two_view_geometries` for the image pair.
    2. Retrieves keypoints indices for the matched points using `pair_id`.
    3. Retrieves the actual coordinates (x1, x2) of the matched keypoints from the `Keypoints` table.
    - Input:
        db_name: Path to the SQLite database.
        image_name1: Name of the first image.
        image_name2: Name of the second image.
    - Output:
        x1: 2D points from the first image (Nx2 numpy array).
        x2: 2D points from the second image (Nx2 numpy array).
    """

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    pair_id_exchange_FLAG = False

    # Retrieve image IDs
    cursor.execute("SELECT image_id FROM Images WHERE name = ?;", (image_name1,))
    image_id1_row = cursor.fetchone()
    if image_id1_row is None:
        raise ValueError(f"Image {image_name1} not found in the database.")
    image_id1 = image_id1_row[0]

    cursor.execute("SELECT image_id FROM Images WHERE name = ?;", (image_name2,))
    image_id2_row = cursor.fetchone()
    if image_id2_row is None:
        raise ValueError(f"Image {image_name2} not found in the database.")
    image_id2 = image_id2_row[0]

    # Retrieve `pair_id` for the image pair from `two_view_geometries`
    pair_id = f"{image_name1}_{image_name2}"
    cursor.execute("SELECT pair_id FROM two_view_geometries WHERE pair_id = ?;", (pair_id,))
    result = cursor.fetchone()

    if result is None:
        # Try with reversed order
        pair_id = f"{image_name2}_{image_name1}"
        cursor.execute("SELECT pair_id FROM two_view_geometries WHERE pair_id = ?;", (pair_id,))
        result = cursor.fetchone()
        pair_id_exchange_FLAG = True

    if result is None:
        conn.close()
        raise ValueError(f"Pair ID not found for images: {image_name1} and {image_name2}")

    # Retrieve matched keypoint indices from the `Matches` table using `pair_id`
    cursor.execute("""
    SELECT KeypointID1, KeypointID2
    FROM Matches
    WHERE pair_id = ?;
    """, (pair_id,))
    matches = cursor.fetchall()

    if not matches:
        raise ValueError(f"No matches found for pair ID: {pair_id}")

    # BUG: SOlucionar cuando se dan vuelta los nombres
    #FIXED?
    if not pair_id_exchange_FLAG:
        keypoint_ids1, keypoint_ids2 = zip(*matches)
    else:
        keypoint_ids2, keypoint_ids1 = zip(*matches)

    # Retrieve keypoints for the first image
    cursor.execute(f"""
    SELECT col, row
    FROM Keypoints
    WHERE ImageID = ? AND KeypointID IN ({','.join('?' * len(keypoint_ids1))});
    """, (image_id1, *keypoint_ids1))
    keypoints1 = cursor.fetchall()

    # Retrieve keypoints for the second image
    cursor.execute(f"""
    SELECT col, row
    FROM Keypoints
    WHERE ImageID = ? AND KeypointID IN ({','.join('?' * len(keypoint_ids2))});
    """, (image_id2, *keypoint_ids2))
    keypoints2 = cursor.fetchall()

    if not keypoints1 or not keypoints2:
        raise ValueError("Keypoints could not be retrieved.")

    conn.close()

    # Convert to numpy arrays
    x1 = np.array(keypoints1, dtype=np.float32)
    x2 = np.array(keypoints2, dtype=np.float32)

    return x1, x2

def build_correspondence_graph(db_name):
    """
    Reads data from the database and constructs:
      1. images_info: a dict storing keypoint locations, etc.
      2. adjacency: a nested dict storing matched keypoints between image pairs.
    Returns (images_info, adjacency).
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # --------------------------------------------------
    # 1. Load images
    # --------------------------------------------------
    images_info = {}
    cursor.execute("SELECT image_id, name, camera_id FROM Images")
    rows = cursor.fetchall()
    for (image_id, name, camera_id) in rows:
        images_info[image_id] = {
            "name": name,
            "camera_id": camera_id,
            "keypoints": {},  # keypoint_id -> (col, row)
            "kp3D": {}        # keypoint_id -> Point3DID or None
        }

    # --------------------------------------------------
    # 2. Load keypoints
    # --------------------------------------------------
    cursor.execute("SELECT ImageID, KeypointID, col, row FROM Keypoints")
    rows = cursor.fetchall()
    for (ImageID, KeypointID, r, c) in rows:
        if ImageID in images_info:
            images_info[ImageID]["keypoints"][KeypointID] = (r, c)

    # --------------------------------------------------
    # 3. Build adjacency structure
    # --------------------------------------------------
    adjacency = defaultdict(lambda: defaultdict(list))
    cursor.execute("""
        SELECT pair_id, ImageID1, KeypointID1, ImageID2, KeypointID2
        FROM Matches
    """)
    rows = cursor.fetchall()
    for (pair_id, img1, kp1, img2, kp2) in rows:
        if img1 in images_info and img2 in images_info:
            adjacency[img1][img2].append((kp1, kp2))
            adjacency[img2][img1].append((kp2, kp1))

    conn.close()
    return images_info, adjacency

def insert_3d_points_in_memory_and_db(db_name, images_info, X_c1, match_list, c1_id, c2_id):
    """
    Inserts triangulated 3D points into Points3D, and updates Tracks for camera 1 & camera 2.
    X_c1: np.array shape (3, N)
    match_list: list of (kp1, kp2) pairs from adjacency
    c1_id, c2_id: image IDs
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    for i in range(X_c1.shape[1]):
        X, Y, Z = X_c1[:3, i]

        kp1, kp2 = match_list[i]

        # ---- 1) CHECK IF ALREADY HAS A 3D ASSIGNMENT (IN-MEMORY) ----
        kp1_3d_id = images_info[c1_id]["kp3D"].get(kp1, None)
        kp2_3d_id = images_info[c2_id]["kp3D"].get(kp2, None)

        # Check in-memory data structure
        if kp1_3d_id is not None or kp2_3d_id is not None:
        # Either keypoint already has a 3D point assigned => skip in-memory and db
            continue

         # Check database
        kp1_3d_id = already_has_3D_in_db(db_name, c1_id, kp1)
        kp2_3d_id = already_has_3D_in_db(db_name, c2_id, kp2)

        #if kp1_3d_id not False
        if kp1_3d_id is not False or kp2_3d_id is not False:

            images_info[c1_id]["kp3D"][kp1] = kp1_3d_id
            images_info[c2_id]["kp3D"][kp2] = kp2_3d_id
            continue

        
        ## IF NOT, insert in db and data-structure in-memory
        # Insert the new 3D point
        cursor.execute("""
            INSERT OR IGNORE INTO Points3D (X, Y, Z, R, G, B, Error)
            VALUES (?, ?, ?, 128, 128, 128, 0.0)
        """, (float(X), float(Y), float(Z)))
        
        # Get the newly assigned Point3DID
        new_3d_id = cursor.lastrowid

        # Add track references: match_list[i] -> (kp1, kp2)
        # Insert track for camera 1
        cursor.execute("""
            INSERT OR IGNORE INTO Tracks (Point3DID, ImageID, KeypointID)
            VALUES (?, ?, ?)
        """, (new_3d_id, c1_id, kp1))

        # Insert track for camera 2
        cursor.execute("""
            INSERT OR IGNORE INTO Tracks (Point3DID, ImageID, KeypointID)
            VALUES (?, ?, ?)
        """, (new_3d_id, c2_id, kp2))

        # Also update  in-memory data structure
        images_info[c1_id]["kp3D"][kp1] = new_3d_id
        images_info[c2_id]["kp3D"][kp2] = new_3d_id

    conn.commit()
    conn.close()

def already_has_3D_in_db(db_name, image_id, kp_id):
    """
    Returns True if (image_id, kp_id) is already in 'Tracks' 
    (meaning that keypoint is assigned to some 3D point).
    """
    import sqlite3
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT Point3DID
        FROM Tracks
        WHERE ImageID = ? AND KeypointID = ?
    """, (image_id, kp_id))
    row = cursor.fetchone()
    conn.close()
    if row is not None:
        return row[0]  # Return the Point3DID
    else:
        return False  # Return False if no row is found

def get_3d_point_coordinates(db_name, point_id):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT X, Y, Z FROM Points3D WHERE Point3DID=?", (point_id,))
    row = cursor.fetchone()
    conn.close()
    return row  # (X, Y, Z)

def find_untriangulated_matches(images_info, adjacency, c1_id, c2_id):
    """
    Return the matched keypoints (kp1, kp2) between c1_id and c2_id
    that do NOT already have a 3D ID assigned (in both memory and DB).
    """
    matches = adjacency[c1_id][c2_id]  # list of (kp1, kp2)
    new_match_list = []
    for (kp1, kp2) in matches:
        kp1_3d = images_info[c1_id]["kp3D"].get(kp1, None)
        kp2_3d = images_info[c2_id]["kp3D"].get(kp2, None)
        if kp1_3d is None and kp2_3d is None:
            new_match_list.append((kp1, kp2))
    return new_match_list

def get_all_3d_points(db_name):
    """
    Fetch all (X, Y, Z) rows from Points3D in the given SQLite database.
    Returns: np.array of shape (N, 3), where N is the number of 3D points.
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()
    cursor.execute("SELECT X, Y, Z FROM Points3D")
    rows = cursor.fetchall()  # list of (X, Y, Z) tuples
    conn.close()

    # Convert to a NumPy array
    points_3d = np.array(rows, dtype=float)  # shape (N, 3)
    return points_3d

def load_observations_and_points(db_name):
    """
    Reads the Tracks table (for 3D assignments) and Keypoints table (for 2D positions)
    from the database, returns:
      - obs_list: a list of (image_id, point3D_id, x, y)
      - points_3d_dict: dict of point3D_id -> (X, Y, Z)
    """
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # 1) Load all 3D points into a dict
    points_3d_dict = {}
    cursor.execute("SELECT Point3DID, X, Y, Z FROM Points3D")
    for (pid, X, Y, Z) in cursor.fetchall():
        points_3d_dict[pid] = [X, Y, Z]   # we can store as list to make them mutable if needed

    # 2) load obs_list from db joining Tracks and Keypoints
    obs_list = []
    cursor.execute("""
        SELECT Tracks.Point3DID, Tracks.ImageID, Tracks.KeypointID, 
               Keypoints.col, Keypoints.row
        FROM Tracks
        JOIN Keypoints ON 
            Tracks.ImageID = Keypoints.ImageID 
            AND Tracks.KeypointID = Keypoints.KeypointID
    """)
    rows = cursor.fetchall()
    for (pid, img_id, kp_id, c, r) in rows:

        x_meas = c
        y_meas = r
        obs_list.append((img_id, pid, x_meas, y_meas))

    conn.close()
    return obs_list, points_3d_dict

#endregion

#################################################### OTHER FUNCTIONS ####################################################
#region 

def save_matrix(file_path, matrix):
    """
    Save a matrix to a file.
    - Input:
        · file_path (str): Path to the file.
        · matrix (np.array): Matrix to save.
    """
    np.savetxt(file_path, matrix)

def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {str(e)}")

def extract_camera_pose(camera_data, camera_id):
    rvec = camera_data[camera_id]["rvec"]
    tvec = camera_data[camera_id]["tvec"]
    R = expm(crossMatrix(rvec))
    T = ensamble_T(R, tvec)
    T_inv = np.linalg.inv(T)
    return T_inv

def extract_rvec_(camera_data, camera_id):
    rvec = camera_data[camera_id]["rvec"]
    return rvec

def extract_tvec(camera_data, camera_id):
    tvec = camera_data[camera_id]["tvec"]
    return tvec

#endregion