#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 1
#
# Title: 2D-3D geometry in homogeneous coordinates and camera projection
#
# Date: 5 September 2024
#
#####################################################################################
#
# Authors:  David Padilla Orenga
#           Nacho Pastore
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
    """
        Load a matrix from a file and handle errors if the file is missing or invalid.
    """
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
    - x1, x2: 2D points from image 1 and 2 (shape: 2 x num_points, where each column is a point)
    - P1, P2: Projection matrices for camera 1 and 2
    """
    num_points = x1.shape[1]  # Number of points is determined by the number of columns
    X_h = np.zeros((4, num_points))  # Homogeneous coordinates for 3D points
    
    for i in range(num_points):
        A = np.zeros((4, 4))
        A[0] = x1[0, i] * P1[2] - P1[0]  # x1[0, i] is the x-coordinate of the i-th point
        A[1] = x1[1, i] * P1[2] - P1[1]  # x1[1, i] is the y-coordinate of the i-th point
        A[2] = x2[0, i] * P2[2] - P2[0]  # x2[0, i] is the x-coordinate of the i-th point in image 2
        A[3] = x2[1, i] * P2[2] - P2[1]  # x2[1, i] is the y-coordinate of the i-th point in image 2
        
        # Solve using SVD
        _, _, Vt = np.linalg.svd(A)
        X_h[:, i] = Vt[-1]
        X_h[:, i] /= X_h[-1, i]  # Normalize to make last coordinate 1 (homogeneous)

    return X_h

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

def plot_epipolar_line(F, x1, ax2, img2, show_epipoles):
    """
    Given a fundamental matrix and a point in image 1, plot the corresponding epipolar line in image 2.
    - F: Fundamental matrix (3x3)
    - x1: Point in image 1 (homogeneous coordinates, shape (3,))
    - ax2: Axis for image 2 where the epipolar line will be plotted
    - img2: Image 2 (for replotting as background)
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
    
    # Clear the axis and re-plot image 2 with the new epipolar line
    # ax2.clear()
    # ax2.imshow(img2)
    ax2.plot(x_vals, y_vals, 'r')  # Plot the epipolar line in red
    ax2.set_title('Image 2 - Epipolar Lines')
    plt.draw()  # Redraw the figure to update the plot

def on_click(event, F, ax1, ax2, img2, show_epipoles):
    """
    Event handler for mouse click. Computes and plots the epipolar line in image 2 based on the click in image 1.
    - event: The mouse event (contains the click coordinates)
    - F: Fundamental matrix
    - ax2: Axis for image 2 where the epipolar line will be plotted
    - img2: Image 2 (for replotting as background)
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
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 fila, 2 columnas
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
    
def compute_essential_matrix_from_R_t(R, t):
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
    - points: Input points (shape: 2 x N, where each column is a point [x, y])
    Returns:
    - points_norm: Normalized points
    - T: Normalization matrix
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
    - x1, x2: Corresponding points from image 1 and image 2 (shape: 2 x N)
    Returns:
    - F: The estimated fundamental matrix
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
    F_norm = Vt[-1].reshape(3, 3)  # The last row of V gives the solution
    
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
    - F: Fundamental matrix (3x3)
    Returns: Epipole in Image 1 and Epipole in Image 2
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
# Compute the Essential Matrix from the Fundamental Matrix
def compute_essential_matrix_from_F(F, K1, K2):
    return K2.T @ F @ K1

# Decompose the Essential Matrix into four possible camera transformations
def decompose_essential_matrix(E):
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

# Check if a set of 3D points are in front of both cameras
def is_valid_pose(X, P1, P2):
    X_cartesian = X[:3, :] / X[3, :]
    depth1 = P1[2, :] @ X
    depth2 = P2[2, :] @ X

    # print(f"Depth values for Camera 1: {depth1}")
    # print(f"Depth values for Camera 2: {depth2}")

    return np.all(depth1 > 0) and np.all(depth2 > 0)

# Select the correct camera pose
def select_correct_pose(x1_h, x2_h, K1, K2, R1, R2, t):

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
    - ax: The 3D plot axis.
    - X: Triangulated 3D points (non-homogeneous coordinates).
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

def to_world_frame (X_h, T_w_c):
    """
    Transform 3D points from camera frame to world frame.
    - X_h: 3D points in camera frame (homogeneous coordinates, shape 4xN).
    - T_w_c: Transformation matrix from camera frame to world frame (4x4 matrix).
    Returns:
    3D points in world frame (homogeneous coordinates, shape 4xN).
    """
    return T_w_c @ X_h


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
        
        R = R2 @ R1.T # Relative rotation
        t = t2 - R2 @ R1.T @ t1 # Relative translation
        E = compute_essential_matrix_from_R_t(R, t)
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

        X_w_correct = to_world_frame(X_correct, T_w_c1)

        print("\n Estimated with 8-point algorithm: \n")
        compute_rmse_in_world_frame(X_w_correct, X_w_sol)
        print("\n Calculated with T_c_w: \n")
        compute_rmse_in_world_frame(X_w_sol, X_h)

        
    except Exception as e:
        print(f"An error occurred: {e}")