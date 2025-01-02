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

#TODO: Need to check the projection of the second camera (or maybe is the plotting)
def count_valid_points(X, P1, P2):
    """
     Count the number of valid points in front of both cameras.
     - Input:
         · X (np.array): Triangulated 3D points (4xN, homogeneous coordinates)
         · P1, P2 (np.array): Projection matrices (3x4)
        - Output:
            · int: Number of valid points
    """
    #DEBUG

    depth1 = P1[2, :] @ X
    depth2 = P2[2, :] @ X
    # valid_depths = (depth1 > 1e-6) & (depth2 > 1e-6)
    valid_depths1 = depth1 > 1e-6
    ##################################### CHECK IF IT IS CORRECT ######################################
    # HARDCODE to negativa, check future if it is correct
    valid_depths2 = depth2 < 1e-6
    #################################################################################################### 
    count_valid_depths1 = np.sum(valid_depths1)
    count_valid_depths2 = np.sum(valid_depths2)
    count_all_valid_depths = count_valid_depths1 + count_valid_depths2
    return count_all_valid_depths
    
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
    #P2_1

    ax = plt.axes(projection='3d', adjustable='box')
    fig3D = draw_possible_poses(ax, R1_t, X1)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_t')
    plt.show()
    #P2_2
    fig3D = draw_possible_poses(ax, R1_minust, X2)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_minust')
    plt.show()
    #P2_3
    fig3D = draw_possible_poses(ax, R2_t, X3)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_t')
    plt.show()
    #P2_4
    fig3D = draw_possible_poses(ax, R2_minust, X4)
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

def select_correct_pose_flexible(x1_h, x2_h, K1, K2, R1, R2, t):
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
    #P2_1
    ax = plt.axes(projection='3d', adjustable='box')
    fig3D = draw_possible_poses(ax, R1_t, X1)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_t')
    plt.show()
    #P2_2
    fig3D = draw_possible_poses(ax, R1_minust, X2)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_minust')
    plt.show()
    #P2_3
    fig3D = draw_possible_poses(ax, R2_t, X3)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_t')
    plt.show()
    #P2_4
    fig3D = draw_possible_poses(ax, R2_minust, X4)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_minust')
    plt.show()


    # Count valid points for each pose
    valid_counts = [
        count_valid_points(X1, P1, P2_1),
        count_valid_points(X2, P1, P2_2),
        count_valid_points(X3, P1, P2_3),
        count_valid_points(X4, P1, P2_4)
    ]
    print("Valid counts")
    print(valid_counts)
    # Select the pose with the maximum valid points
    best_pose_idx = np.argmax(valid_counts)
    
    #DEBUG:
    # best_pose_idx = 1

    poses = [(R1, t, X1), (R1, -t, X2), (R2, t, X3), (R2, -t, X4)]

    if valid_counts[best_pose_idx] == 0:
        raise ValueError("No valid pose found!")
    print("Best_pose_idx")
    print(best_pose_idx)
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

#endregion

    
#################################################### PLOTTING FUNCTIONS ####################################################
#region Plotting FUNCTIOS

def draw_possible_poses(ax, Rt, X):
    fig3D = plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    drawRefSystem(ax, np.eye(4, 4), '-', 'C1')
    drawRefSystem(ax, Rt, '-', 'C2')
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

def visualize_residuals(image, observed_points, projected_points, title, ax=None):
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
    
    ax.set_xlim([0, image.shape[1]])
    ax.set_ylim([image.shape[0], 0])
    
    ax.set_title(title)
    ax.legend()

def visualize_3D_w_2cameras(T_w_c1, T_w_c2, X_w):
    """
    Plot the 3D cameras and the 3D points.
    """
    plt.figure()
    ax = plt.axes(projection='3d', adjustable='box')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    drawRefSystem(ax, T_w_c1, '-', 'C1')
    drawRefSystem(ax, T_w_c2, '-', 'C2')

    ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    # plotNumbered3DPoints(ax, X_w, 'r', 0.1)

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

def visualize_3D_comparison(ax, T_c1_c2, X_c1_w, T_c2_c1_initial, X_c1_w_initial, T_c2_c1_opt, X_c1_w_opt):
    """
    Plot the 3D cameras and the 3D points for ground truth, initial guess, and optimized solution.
    - Ground truth: uses T_w_c1, T_w_c2, X_w
    - Initial: uses T_c2_c1_initial, X_c1_w_initial
    - Optimized: uses T_c2_c1_opt, X_c1_w_opt
    """
    # Ground Truth (World Frame)
    drawRefSystem(ax, np.eye(4), '-', 'C1 (Ground Truth)')
    drawRefSystem(ax, T_c1_c2, '-', 'C2 (Ground Truth)')
    ax.scatter(X_c1_w[0, :], X_c1_w[1, :], X_c1_w[2, :], marker='o', color='g', label='3D Points (Ground Truth)')

    # Initial Guess (Camera 1 Frame)
    drawRefSystem(ax, T_c2_c1_initial, '--', 'C2 (Initial)')
    ax.scatter(X_c1_w_initial[0, :], X_c1_w_initial[1, :], X_c1_w_initial[2, :], marker='^', color='b', label='3D Points (Initial)')

    # Optimized Solution (Camera 1 Frame)
    drawRefSystem(ax, T_c2_c1_opt, '-.', 'C2 (Optimized)')
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
    ax1.set_title('Image 1 - Select Point')
    
    # Segundo subplot para la segunda imagen
    ax2.set_xlabel('Coordinates X (píxeles)')
    ax2.set_ylabel('Coordinates Y (píxeles)')
    ax2.imshow(img2)
    ax2.set_title('Image 2 - Epipolar Lines')

    # Connect the click event on image 1 to the handler
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, F , ax1, ax2, img2,show_epipoles))
    print('\nClose the figure to continue. Select a point from Img1 to get the equivalent epipolar line.')
    plt.show()


#endregion

   
#################################################### BUNDLE FUNCTIONS ####################################################
#region BUNDLE FUNCTIONS

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints):
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
 
    theta = Op[:3]              # Rotation vector
    # t = Op[3:6]
    # Translation vector
    t_theta = Op[3]
    t_phi = Op [4]
    t = np.array([np.sin(t_theta)*np.cos(t_phi), np.sin(t_theta)*np.sin(t_phi), np.cos(t_theta)])
        

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

#endregion


#################################################### SQLITE FUNCTIONS ####################################################
#region SQLITE FUNCTIONS

def extract_R_t_from_F(db_name, image_name1, image_name2, K):

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

    # Retrieve the F matrix for the image pair
    pair_id = f"{image_name1}_{image_name2}"
    cursor.execute("SELECT F FROM two_view_geometries WHERE pair_id = ?;", (pair_id,))
    result = cursor.fetchone()

    if result is None:
        # Try with the names interchanged
        pair_id = f"{image_name2}_{image_name1}"
        cursor.execute("SELECT F FROM two_view_geometries WHERE pair_id = ?;", (pair_id,))
        result = cursor.fetchone()

    if result is None:
        conn.close()
        raise ValueError(f"Fundamental matrix not found for pair: {image_name1} and {image_name2}")

    
    F_json = result[0]
    F = np.array(json.loads(F_json))
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
    import sqlite3
    import numpy as np

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

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

    keypoint_ids1, keypoint_ids2 = zip(*matches)

    # Retrieve keypoints for the first image
    cursor.execute(f"""
    SELECT row, col
    FROM Keypoints
    WHERE ImageID = ? AND KeypointID IN ({','.join('?' * len(keypoint_ids1))});
    """, (image_id1, *keypoint_ids1))
    keypoints1 = cursor.fetchall()

    # Retrieve keypoints for the second image
    cursor.execute(f"""
    SELECT row, col
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
#endregion

