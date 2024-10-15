#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors:  David Padilla Orenga 946874
#           Nacho Pastore Benaim 920576
#
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

def get_projection_matrix(K, T):
    """
        Computes the projection matrix (3x3) from camera matrix (K) and transformation matrix (T).
        - Inputs:
            · K (np.array): Camera intrinsic matrix (3x3).
            · T (np.array): Transformation matrix (4x4).
        - Output:
            · np.array: Projection matrix.
    """
    Rt = T[:3]
    P = K @ Rt
    return P

def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {str(e)}")

def project_to_camera(X_h, P):
    x_h_projected = P @ X_h
    x_h_projected /= x_h_projected[-1]
    return x_h_projected

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

def drawRefSystem(ax, T_w_c, strStyle, nameStr):
    """
        Draw a reference system in a 3D plot: Red for X axis, Green for Y axis, and Blue for Z axis
        -Input:
            · ax: axis handle
            · T_w_c (np.array): (4x4 matrix) Reference system C seen from W.
            · strStyle: lines style.
            · nameStr: Name of the reference system.
    """
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 0:1], strStyle, 'r', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 1:2], strStyle, 'g', 1)
    draw3DLine(ax, T_w_c[0:3, 3:4], T_w_c[0:3, 3:4] + T_w_c[0:3, 2:3], strStyle, 'b', 1)
    ax.text(np.squeeze( T_w_c[0, 3]+0.1), np.squeeze( T_w_c[1, 3]+0.1), np.squeeze( T_w_c[2, 3]+0.1), nameStr)
    
def draw3DLine(ax, xIni, xEnd, strStyle, lColor, lWidth):
    """
        Draw a segment in a 3D plot
        -Input:
            · ax: axis handle
            · xIni: Initial 3D point.
            · xEnd: Final 3D point.
            · strStyle: Line style.
            · lColor: Line color.
            · lWidth: Line width.
    """
    ax.plot([np.squeeze(xIni[0]), np.squeeze(xEnd[0])], [np.squeeze(xIni[1]), np.squeeze(xEnd[1])], [np.squeeze(xIni[2]), np.squeeze(xEnd[2])],
            strStyle, color=lColor, linewidth=lWidth)

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

def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector v.
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])
    
def     compute_essential_matrix_from_R_t(R, t):
    """
    Computes the essential matrix E given rotation R and translation t.
    """
    return skew_symmetric(t) @ R

def compute_fundamental_matrix(E, K1, K2):
    """
    Computes the fundamental matrix F given essential E and tintrinsic matrices K1 and K2.
    """
    return np.linalg.inv(K2).T @ E @ np.linalg.inv(K1)

def normalize_points(points):
    """
        Normalize a set of points for the eight-point algorithm to improve numerical stability.
        - Input:
            · points: Input points (shape: 2 x N, where each column is a point [x, y])
        - Output:
            · points_norm: Normalized points
            · T: Normalization matrix
    """
    mean = np.mean(points, axis=1)
    std = np.std(points, axis=1)
    
    # Construct normalization matrix
    T = np.array([[1/std[0], 0, -mean[0]/std[0]],
                  [0, 1/std[1], -mean[1]/std[1]],
                  [0, 0, 1]])
    
    # Convert points to homogeneous coordinates
    points_hom = np.vstack((points, np.ones((1, points.shape[1]))))
    
    # Apply the normalization
    points_norm = T @ points_hom
    return points_norm, T

def eight_point_algorithm(x1, x2):
    """
        Compute the fundamental matrix using the eight-point algorithm.
        - Input:
            · x1, x2: Corresponding points from image 1 and image 2 (shape: 2 x N)
        -Output:
            · F: The estimated fundamental matrix
    """
    # Normalize the points
    x1_norm, T1 = normalize_points(x1)
    x2_norm, T2 = normalize_points(x2)
    
    # Construct the matrix A based on the normalized points
    N = x1.shape[1]
    A = np.zeros((N, 9))
    for i in range(N):
        A[i] = [
            x2_norm[0, i] * x1_norm[0, i],  # x2' * x1'
            x2_norm[0, i] * x1_norm[1, i],  # x2' * y1'
            x2_norm[0, i],                  # x2'
            x2_norm[1, i] * x1_norm[0, i],  # y2' * x1'
            x2_norm[1, i] * x1_norm[1, i],  # y2' * y1'
            x2_norm[1, i],                  # y2'
            x1_norm[0, i],                  # x1'
            x1_norm[1, i],                  # y1'
            1
        ]
    
    # Solve Af = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    F_norm = Vt[-1].reshape(3, 3)
    
    # Enforce the rank-2 constraint on F (set the smallest singular value to 0)
    U, S, Vt = np.linalg.svd(F_norm)
    S[-1] = 0
    F_norm = U @ np.diag(S) @ Vt
    
    # Denormalize the fundamental matrix
    F = T2.T @ F_norm @ T1
    return F

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
    X_cartesian = X[:3, :] / X[3, :]
    depth1 = P1[2, :] @ X
    depth2 = P2[2, :] @ X

    return np.all(depth1 > 0) and np.all(depth2 > 0)

def select_correct_pose(x1_h, x2_h, K1, K2, R1, R2, t):
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
        raise ValueError("No valid pose found!")
    
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
    
def compute_rmse_in_world_frame(X_w_correct, X_w_sol):
    """
    Compute RMSE between estimated points in World frame and ground truth points in world frame.
    X_w_correct: Estimated 3D points in World frame (homogeneous coordinates, shape 4xN).
    X_w_sol: Ground truth 3D points in world frame (homogeneous coordinates, shape 4xN).
    """


    # Convert homogeneous to Cartesian coordinates
    X_w_correct_cartesian = X_w_correct[:3, :] / X_w_correct[3, :]
    X_w_sol_cartesian = X_w_sol[:3, :] / X_w_sol[3, :]

    # Compute RMSE
    diff = X_w_correct_cartesian - X_w_sol_cartesian
    rmse = np.sqrt(np.mean(np.sum(diff ** 2, axis=0)))

    print(f"RMSE : {rmse:.4f}")
    return rmse

def change_frame (X_h, T_w_c):
    """
    Transform 3D points from camera frame to world frame.
    - X_h: 3D points in camera frame (homogeneous coordinates, shape 4xN).
    - T_w_c: Transformation matrix from camera frame to world frame (4x4 matrix).
    Returns:
    3D points in world frame (homogeneous coordinates, shape 4xN).
    """
    return T_w_c @ X_h


def compute_homography_from_camera_posse_and_plane(R_c2_c1, t_c2_c1, K1, K2, Pi_c1):
    """
    Args:
        - R_c2_c1: Relative rotation matrix from Camera 1 to Camera 2 (3x3).
        - t_c2_c1: Relative translation vector from Camera 1 to Camera 2 (3x1).
        - K1: Intrinsic matrix of Camera 1 (3x3).
        - K2: Intrinsic matrix of Camera 2 (3x3).
        - Pi_1: Plane equation coefficients in Camera 1's frame (4x1) [n_x, n_y, n_z, d].

    Returns:
        - H21: Homography matrix (3x3).
    """
    # Ensure all inputs are numpy arrays
    R_c2_c1 = np.array(R_c2_c1)
    t_c2_c1 = np.array(t_c2_c1)
    K1 = np.array(K1)
    K2 = np.array(K2)
    Pi_c1 = np.array(Pi_c1)

    # Ensure Pi_1 is a column vector (4x1)
    Pi_c1 = Pi_c1.reshape(4, 1)

    # Ensure t_c2_c1 is a column vector (3x1)
    t_c2_c1 = t_c2_c1.reshape(3, 1)

    # Separate the normal vector (n1) and distance (d) from the plane equation
    n1 = Pi_c1[:3]  # Normal vector to the plane in Camera 1's frame (3x1)
    d = Pi_c1[3, 0]  # Distance from Camera 1 to the plane (scalar)

    # Ensure n1 is a row vector (1x3)
    n1 = n1.reshape(1, 3)

    # Compute the homography matrix
    H21 = K2 @ (R_c2_c1 - (t_c2_c1 @ n1) / d) @ np.linalg.inv(K1)
    return H21

def on_click_homography(event, H21, ax1, ax2, img2):
    """
    Event handler for mouse click. Transfers the clicked point from Image 1 to Image 2 using the homography matrix.
    - event: The mouse event (contains the click coordinates)
    - H21: Homography matrix that transfers points from Image 1 to Image 2 (3x3)
    - ax2: Axis for image 2 where the transferred point will be plotted
    - img2: Image 2 (for replotting as background)
    """
    # Get the click coordinates in Image 1
    x_clicked = event.xdata
    y_clicked = event.ydata

    # Mark the clicked point in image 1 with an "x"
    ax1.plot(x_clicked, y_clicked, 'rx', markersize=10)  # Red "x" at the clicked point
    ax1.set_title('Image 1 - Select Point')

    if x_clicked is not None and y_clicked is not None:
        print(f"Clicked coordinates in Image 1: ({x_clicked:.2f}, {y_clicked:.2f})")
        
        # Convert the clicked point to homogeneous coordinates
        x1_clicked = np.array([x_clicked, y_clicked, 1]).reshape(3, 1)  # 3x1 column vector
        
        # Transfer the point to Image 2 using the homography H21
        x2_transferred = H21 @ x1_clicked  # Apply homography
        x2_transferred /= x2_transferred[2]  # Normalize homogeneous coordinates
        
        # Extract x and y from the transferred point
        x2_x = x2_transferred[0, 0]
        x2_y = x2_transferred[1, 0]
        
        # Plot the transferred point in Image 2
        ax2.plot(x2_x, x2_y, 'rx', markersize=10)  # Plot blue circle at transferred point
        ax2.set_title('Image 2 - Transferred Points From Homography')
        plt.draw()  # Redraw the figure to update the plot

def visualize_point_transfer_from_homography(img1, img2, H21):
    """
    Visualize point transfer between Image 1 and Image 2 using a homography matrix H21.
    Clicking on Image 1 transfers the point to Image 2.
    - img1: Image 1 where points are clicked.
    - img2: Image 2 where transferred points are plotted.
    - H21: Homography matrix (3x3).
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))

    # Configure the first subplot (Image 1)
    ax1.set_xlabel('Coordinates X (pixels)')
    ax1.set_ylabel('Coordinates Y (pixels)')
    ax1.imshow(img1)
    ax1.set_title('Image 1 - Click Points')

    # Configure the second subplot (Image 2)
    ax2.set_xlabel('Coordinates X (pixels)')
    ax2.set_ylabel('Coordinates Y (pixels)')
    ax2.imshow(img2)
    ax2.set_title('Image 2 - Transferred Points From Homography')

    # Connect the click event on image 1 to the handler
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click_homography(event, H21, ax1, ax2, img2))
    
    print('\nClose the figure to continue. Select a point from Img1 to get the equivalent transferred point in Img2.')
    plt.show()

def estimate_homography_from_points(x1, x2):
    """
    Estimate the homography matrix from a set of point correspondences using DLT (Direct Linear Transform).
    Args:
    - x1: Points in Image 1 (shape: 2 x N or 3 x N, where N is the number of points).
    - x2: Corresponding points in Image 2 (shape: 2 x N or 3 x N).
    
    Returns:
    - H21: Estimated homography matrix (3x3) from image 1 to image 2.
    """
    N = x1.shape[1]
    if N < 4:
        raise ValueError("At least 4 point correspondences are required to estimate the homography.")

    # Ensure points are in homogeneous coordinates (3xN)
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, N))))
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, N))))
    
    # Build the system of equations
    A = []
    for i in range(N):
        x1i, y1i, _ = x1[:, i]
        x2i, y2i, _ = x2[:, i]
        A.append([-x1i, -y1i, -1, 0, 0, 0, x2i * x1i, x2i * y1i, x2i])
        A.append([0, 0, 0, -x1i, -y1i, -1, y2i * x1i, y2i * y1i, y2i])

    A = np.array(A)

    # Solve Ah = 0 using SVD
    _, _, Vt = np.linalg.svd(A)
    H21 = Vt[-1].reshape(3, 3)

    # Normalize the homography so that H[2,2] = 1
    H21 /= H21[2, 2]

    return H21

def compute_rmse_of_homography_with_ground_truth(H21, x1, x2):
    """
    Compute the RMSE between the projected points (using the homography) and the actual points in Image 2.
    - H21: Estimated homography matrix (3x3).
    - x1: Points in Image 1 (3xN, homogeneous coordinates).
    - x2: Points in Image 2 (3xN, homogeneous coordinates).
    
    Returns:
    - RMSE: Root Mean Squared Error in the image plane.
    """
    # Number of points
    N = x1.shape[1]

    # Ensure points are in homogeneous coordinates (3xN)
    if x1.shape[0] == 2:
        x1 = np.vstack((x1, np.ones((1, N))))
    if x2.shape[0] == 2:
        x2 = np.vstack((x2, np.ones((1, N))))

    # Project points from Image 1 to Image 2 using the homography
    projected_x2 = H21 @ x1  # Apply homography to points in Image 1
    projected_x2 /= projected_x2[2]  # Normalize homogeneous coordinates

    # Compute squared differences (x and y coordinates only)
    diff = projected_x2[:2] - x2[:2]  # Ignore the homogeneous component for RMSE
    squared_diff = np.sum(diff**2, axis=0)

    # Compute the RMSE
    rmse = np.sqrt(np.mean(squared_diff))
    print(f"RMSE: {rmse:.4f}")
    return rmse

if __name__ == '__main__':
    
    np.set_printoptions(precision=1,linewidth=1024,suppress=True)
    
    try:
        #region
        # Initialize camera and world parameters
        T_w_c1 = load_matrix('./T_w_c1.txt')
        T_w_c2 = load_matrix('./T_w_c2.txt')
        T_c1_w = np.linalg.inv(T_w_c1)
        T_c2_w = np.linalg.inv(T_w_c2)
        K_C = load_matrix('./K_c.txt')  

        # Compute projection matrices
        P1 = get_projection_matrix(K_C, T_c1_w)
        P2 = get_projection_matrix(K_C, T_c2_w)
        
        # Loading set of points already projected
        x1_data = load_matrix('./x1Data.txt')  # 2D points from image 1
        x2_data = load_matrix('./x2Data.txt')  # 2D points from image 2
        X_h = triangulate_points(x1_data, x2_data, P1, P2)
        X_w_sol = load_matrix('./X_w.txt')

        print("\nProjection Matrix P1:")
        print(P1)
        print("\nProjection Matrix P2:")
        print(P2)
        
        print("\nProjection Matrix x_H:")
        print(X_h[:,0:3])


        ##Plot the 3D cameras and the 3D points
        fig3D = plt.figure(3)

        ax = plt.axes(projection='3d', adjustable='box')
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_zlabel('Z')

        drawRefSystem(ax, np.eye(4, 4), '-', 'W')
        drawRefSystem(ax, T_w_c1, '-', 'C1')
        drawRefSystem(ax, T_w_c2, '-', 'C2')
        
        ax.scatter(X_h[0, :], X_h[1, :], X_h[2, :], marker='.', color='r')
        ax.scatter(X_w_sol[0, :], X_w_sol[1, :], X_w_sol[2, :], marker='.', color='b')
        
        #Matplotlib does not correctly manage the axis('equal')
        xFakeBoundingBox = np.linspace(0, 4, 2)
        yFakeBoundingBox = np.linspace(0, 4, 2)
        zFakeBoundingBox = np.linspace(0, 4, 2)
        plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')
        print('\nClose the figure to continue. Left button for orbit, right button for zoom.')
        plt.show()
        #endregion
        
        ## SECTION 2 ##
        F_21_test = load_matrix('./F_21_test.txt')
        
        ## 2D plotting example
        img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
        
        visualize_epipolar_lines(F_21_test, img1, img2, show_epipoles=False)
        
        ## 2.2 CALULATE THE ESSENTIAL AND FUNDAMENTAL MATRICES
        R1 = T_c1_w[:3, :3]
        R2 = T_c2_w[:3, :3]
        t1 = T_c1_w[:3, 3]
        t2 = T_c2_w[:3, 3]
        
        R_c2_c1 = R2 @ R1.T # Relative rotation
        t_c2_c1 = t2 - R2 @ R1.T @ t1 # Relative translation
        E = compute_essential_matrix_from_R_t(R_c2_c1, t_c2_c1)
        F = compute_fundamental_matrix(E, K_C, K_C)
        
        print("\nEssential Matrix E:\n", E)
        print("\nCalculated Fundamental Matrix F:\n", F)
        
        visualize_epipolar_lines(F, img1, img2, show_epipoles=True)


       ### 2.3 CALCULATE F USING 8 POINTS ###
        
        x1 = load_matrix('./x1Data.txt')  
        x2 = load_matrix('./x2Data.txt')  
        F_est = eight_point_algorithm(x1, x2)  
        print("\nEstimated Fundamental Matrix F:\n", F_est)
        visualize_epipolar_lines(F_est, img1, img2, show_epipoles=True)

        #### 2.4 and 2.5 POSE ESTIMATION FROM TWO VIEWS ###

        x1_h = np.vstack((x1, np.ones(x1.shape[1])))
        x2_h = np.vstack((x2, np.ones(x2.shape[1]))) 

        K1 = K_C
        K2 = K_C

        E = compute_essential_matrix_from_F(F_est, K1, K2)

        R1, R2, t = decompose_essential_matrix(E)

        R_correct, t_correct, X_correct = select_correct_pose(x1_h, x2_h, K1, K2, R1, R2, t)

        print("Correct Rotation Matrix:\n", R_correct)
        print("Correct Translation Vector:\n", t_correct)

        X_w_correct = change_frame(X_correct, T_w_c1)

        print("\n Estimated with 8-point algorithm: \n")
        compute_rmse_in_world_frame(X_w_correct, X_w_sol)
        print("\n Calculated with T_c_w: \n")
        compute_rmse_in_world_frame(X_w_sol, X_h)

        #### SECTION 3- Homography ####

        ## 3.1 Homography from relative poses and common plane ##

        # Load the points from the common plane in Camera 1 frame
        Pi_c1 = load_matrix('./Pi_1.txt')
        H21 = compute_homography_from_camera_posse_and_plane(R_c2_c1, t_c2_c1, K_C, K_C, Pi_c1)

        ## 3.2 Visualize the homography ##

        visualize_point_transfer_from_homography(img1, img2, H21)

        ## 3.3 Compute Homography from matches"
        x1_floor = load_matrix('./x1FloorData.txt')
        x2_floor = load_matrix('./x2FloorData.txt')

        H_21_est = estimate_homography_from_points(x1_floor, x2_floor)
        visualize_point_transfer_from_homography(img1, img2, H_21_est)
        print("Rmse of the H_21_est:")
        compute_rmse_of_homography_with_ground_truth(H_21_est, x1_floor, x2_floor)

        
    except Exception as e:
        print(f"An error occurred: {e}")
