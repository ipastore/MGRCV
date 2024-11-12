import numpy as np
import matplotlib.pyplot as plt
import cv2
import os
from scipy.linalg import expm, logm
from scipy.optimize import least_squares
from mpl_toolkits.mplot3d import Axes3D
import csv
import scipy as sc
import scipy.io as sio



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
    
def skew_symmetric(v):
    """
    Returns the skew-symmetric matrix of a vector v.
    """
    return np.array([[0, -v[2], v[1]],
                     [v[2], 0, -v[0]],
                     [-v[1], v[0], 0]])

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

    return np.all(depth1 > 0) and np.all(depth2 > 0)

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
    
    # ax = plt.axes(projection='3d', adjustable='box')
    # fig3D = draw_possible_poses(ax, R1_t, X1)
    # adjust_plot_limits(ax, X1)
    # plt.title('Possible Pose R1_t')
    # plt.show()
    # #P2_2
    # fig3D = draw_possible_poses(ax, R1_minust, X2)
    # adjust_plot_limits(ax, X1)
    # plt.title('Possible Pose R1_minust')
    # plt.show()
    # #P2_3
    # fig3D = draw_possible_poses(ax, R2_t, X3)
    # adjust_plot_limits(ax, X1)
    # plt.title('Possible Pose R2_t')
    # plt.show()
    # #P2_4
    # fig3D = draw_possible_poses(ax, R2_minust, X4)
    # adjust_plot_limits(ax, X1)
    # plt.title('Possible Pose R2_minust')
    # plt.show()

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
    P2 = get_projection_matrix(K_C, T_21)
    P3 = get_projection_matrix(K_C, T_31)


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

def resBundleProjection3Views8DofChained(Op, x1Data, x2Data, x3Data, K_c, nPoints):
    """
    Residual function for bundle adjustment across three views.

    - Op: Optimization parameters (rotation and translation for cameras 2 and 3, 3D points).
    - x1Data, x2Data, x3Data: 2D points on images 1, 2, and 3.
    - K_c: Intrinsic calibration matrix.
    - nPoints: Number of points.

    - Output: residuals between observed 2D points and projected points from 3D points.
    """
    # Extract parameters for camera 2 (6 DoF)
    theta_12 = Op[:3]
    t_12 = Op[3:6].reshape(3, 1)
    
    # Extract parameters for camera 3 (3 DoF)
    # Assuming rotation around y-axis and translation along z-axis only
    angle_y_23 = Op[6]             # Single angle for rotation around y-axis
    t_z_23 = Op[7]                 # Single translation along z-axis

    # Construct rotation matrix for Camera 2
    R_12 = expm(crossMatrix(theta_12))

    theta_23 = np.array([0, angle_y_23, 0])  # Rotation around y-axis only
    R_23 = expm(crossMatrix(theta_23))

    # Construct translation vectors
    t_23 = np.array([0, 0, t_z_23]).reshape(3,1)  # Only translation along the z-axis

    T_23 = ensamble_T(R_23, t_23)
    T_12 = ensamble_T(R_12, t_12) 

    T_13 = T_12 @ T_23

    # Projection matrices
    P1 = K_c @ np.hstack((np.eye(3), np.zeros((3, 1))))  # Identity for camera 1
    P2 = get_projection_matrix(K_c, T_12)
    P3 = get_projection_matrix(K_c, T_13)


    # Convert 3D points to homogeneous coordinates
    X = Op[8:].reshape(3,-1)
    X_h = np.vstack((X, np.ones((1, nPoints))))

    # Project points to each camera
    x1_projected = project_to_camera(P1, X_h)
    x2_projected = project_to_camera(P2, X_h)
    x3_projected = project_to_camera(P3, X_h)

    # Calculate residuals for each view
    res_x1 = x1Data[:2, :] - x1_projected[:2]
    res_x2 = x2Data[:2, :] - x2_projected[:2]
    res_x3 = x3Data[:2, :] - x3_projected[:2]

    # Flatten residuals and combine them
    residuals = np.hstack((res_x1.flatten(), res_x2.flatten(), res_x3.flatten()))

    return residuals

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
    ax.set_title(title)
    ax.legend()

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

def get_scale_factor(X_ground_truth, X_optimized):
    """
    Compute the scale factor to align the optimized 3D points with the ground truth 3D points.

    Parameters:
        X_ground_truth (np.array): Ground truth 3D points (4xN, homogeneous coordinates).
        X_optimized (np.array): Optimized 3D points (4xN, homogeneous coordinates).

    Returns:
        float: Scale factor.
    """
    # Compute the mean Euclidean norm (distance from the origin) of the ground truth points
    mean_distance_ground_truth = np.mean(np.linalg.norm(X_ground_truth[:3, :], axis=0))

    # Compute the mean Euclidean norm of the optimized points
    mean_distance_optimized = np.mean(np.linalg.norm(X_optimized[:3, :], axis=0))

    # Calculate the scale factor
    scale_factor = mean_distance_ground_truth / mean_distance_optimized

    return scale_factor




if __name__ == '__main__':
    
    np.set_printoptions(precision=1,linewidth=1024,suppress=True)

        # Load the data
    T_w_c1 = load_matrix('../data/T_w_c1.txt')
    T_w_c2 = load_matrix('../data/T_w_c2.txt')
    T_w_c3 = load_matrix('../data/T_w_c3.txt')
    T_c1_w = np.linalg.inv(T_w_c1)
    T_c2_w = np.linalg.inv(T_w_c2)
    T_c3_w = np.linalg.inv(T_w_c3)
    K_C = load_matrix('../data/K_c.txt')  
    x1_data = load_matrix('../data/x1Data.txt')  
    x2_data = load_matrix('../data/x2Data.txt')
    x3_data = load_matrix('../data/x3Data.txt')
    X_w = load_matrix('../data/X_w.txt')
    F_c2_c1 = load_matrix('../data/F_21.txt')
    image1 = cv2.imread('../data/image1.png')
    image2 = cv2.imread('../data/image2.png')
    image3 = cv2.imread('../data/image3.png')
    
    excercise2= True
    excercise3 = True
    excercise4 = True
    excercise4_full = True

    if excercise2:
        # Initial projections using ground truth
        P_c1_w_gt = get_projection_matrix(K_C, T_c1_w)
        P_c2_w_gt = get_projection_matrix(K_C, T_c2_w)
        P_c3_w_gt = get_projection_matrix(K_C, T_c3_w)


        x1_proj_gt = project_to_camera(P_c1_w_gt, X_w)
        x2_proj_gt = project_to_camera(P_c2_w_gt, X_w)
        x3_proj_gt = project_to_camera(P_c3_w_gt, X_w)
        
        # Visualize initial residuals for both views
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        visualize_residuals(image1, x1_data, x1_proj_gt, 'Residuals in Image 1', ax=axs[0])
        visualize_residuals(image2, x2_data, x2_proj_gt, 'Residuals in Image 2', ax=axs[1])
        visualize_residuals(image3, x3_data, x3_proj_gt, 'Residuals in Image 3', ax=axs[2])
        plt.tight_layout()
        plt.show()

        # X data to homogeneous coordinates
        x1_h = np.vstack((x1_data, np.ones(x1_data.shape[1])))
        x2_h = np.vstack((x2_data, np.ones(x2_data.shape[1])))

        # Compute Essential Matrix from F
        E_c2_c1 = compute_essential_matrix_from_F(F_c2_c1, K_C, K_C)
        # Decompose Essential Matrix
        R1, R2, t = decompose_essential_matrix(E_c2_c1)

        R_c2_c1_initial, t_c2_c1_initial, X_c1_w_initial = select_correct_pose(x1_h, x2_h, K_C, K_C, R1, R2, t)
        save_matrix('../data/cache/R_c2_c1_initial.txt', R_c2_c1_initial)
        save_matrix('../data/cache/t_c2_c1_initial.txt', t_c2_c1_initial)
        save_matrix('../data/cache/X_c1_w_initial.txt', X_c1_w_initial)

        # View residuals from initial calculations
        #From World to image1
        P_c1_c1_initial = get_projection_matrix(K_C, np.eye(4))
        x1_proj_initial = project_to_camera(P_c1_c1_initial, X_c1_w_initial)

        #From World to image2
        T_c2_c1_initial = ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
        P_c2_c1_initial = get_projection_matrix(K_C,T_c2_c1_initial)
        x2_proj_initial = project_to_camera(P_c2_c1_initial, X_c1_w_initial)

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(image1, x1_data, x1_proj_initial, "Initial Residuals in Image 1", ax=axs[0])
        visualize_residuals(image2, x2_data, x2_proj_initial, 'Initial Residuals in Image 2', ax=axs[1])
        plt.tight_layout()
        plt.show()

        ## Optimization
        # Initialize the initial guess with correct values
        theta_c2_c1_initial = crossMatrixInv(logm(R_c2_c1_initial.astype('float64')))
        save_matrix('../data/cache/theta_c2_c1_initial.txt', theta_c2_c1_initial)

        # intial_guess = np.hstack((theta_c2_c1_initial, t_c2_c1_initial, X_c1_w_initial[:3, :].flatten())) 
        
        t_norm = np.linalg.norm(t_c2_c1_initial, axis=-1)
        t_theta = np.arccos(t[2]/t_norm)
        t_phi = np.arctan2(t[1], t[0])


        intial_guess = np.hstack((theta_c2_c1_initial, t_theta, t_phi, X_c1_w_initial[:3, :].flatten())) 
        

        nPoints = X_c1_w_initial.shape[1]

        # Perform bundle adjustment
        # optimized = least_squares(resBundleProjection, intial_guess, args=(x1_data, x2_data, K_C, nPoints), max_nfev=10000)
        # optimized = least_squares(resBundleProjection, 
        #                         intial_guess, 
        #                         args=(x1_data, x2_data, K_C, nPoints),
        #                         method='trf', 
        #                         jac='3-point', 
        #                         loss='huber',
        #                         verbose=2)
        
        optimized = least_squares(resBundleProjection, 
                        intial_guess, 
                        args=(x1_data, x2_data, K_C, nPoints),
                        method='lm'
                        )
        



        # Extract optimized parameters
        theta_c2_c1_opt = optimized.x[:3]
        # t_c2_c1_opt = optimized.x[3:5]
        t_theta = optimized.x[3]
        t_phi = optimized.x[4]
        t_c2_c1_opt = np.array([np.sin(t_theta)*np.cos(t_phi), np.sin(t_theta)*np.sin(t_phi), np.cos(t_theta)])
        X_c1_w_opt = optimized.x[5:].reshape(3, -1).T

        # Save X_c1_w_opt_h to a file
        save_matrix('../data/cache/X_c1_w_opt.txt', X_c1_w_opt)

        print("Initial Rotation Vector:\n", theta_c2_c1_initial)
        print("Optimized Rotation Vector:\n", theta_c2_c1_opt)
        print("Initial Translation Vector:\n", t_c2_c1_initial)
        print("Optimized Translation Vector:\n", t_c2_c1_opt)
        
        ## Visualize optimization
        R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
        T_c2_c1_opt = ensamble_T(R_c2_c1_opt, t_c2_c1_opt)
        P_c2_c1_opt = get_projection_matrix(K_C, T_c2_c1_opt)

        X_c1_w_opt_h = np.vstack((X_c1_w_opt.T, np.ones((1, X_c1_w_opt.shape[0]))))

        x1_proj_opt = project_to_camera(P_c1_c1_initial, X_c1_w_opt_h)
        x2_proj_opt = project_to_camera(P_c2_c1_opt, X_c1_w_opt_h)

        # View residuals from optimized calculations
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(image1, x1_data, x1_proj_opt, "Optimized Residuals in Image 1", ax=axs[0])
        visualize_residuals(image2, x2_data, x2_proj_opt, 'Optimized Residuals in Image 2', ax=axs[1])
        plt.tight_layout()
        plt.show()


        # Ground truth
        T_c1_c2_gt = T_c1_w @ T_w_c2
        X_c1_w = T_c1_w @ X_w
        
        # Initial
        T_c1_c2_initial = np.linalg.inv(T_c2_c1_initial)
        
        #Opt
        T_c1_c2_opt = np.linalg.inv(T_c2_c1_opt)



        #Scale
        # scale_factor = get_scale_factor(X_c1_w, X_c1_w_opt)
        # X_c1_w_opt_scaled = X_c1_w_opt * scale_factor
        # X_c1_w_opt_scaled_h = np.vstack((X_c1_w_opt_scaled.T, np.ones((1, X_c1_w_opt_scaled.shape[0]))))
        # T_c1_c2_opt_scaled = T_c1_c2_opt * scale_factor

        # Create a 3D plot to compare all results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        visualize_3D_comparison(
        ax,
        T_c1_c2_gt, X_c1_w,             # Ground truth
        T_c1_c2_initial, X_c1_w_initial, # Initial estimate
        T_c1_c2_opt, X_c1_w_opt_h          # Optimized solution
    )

            
    if excercise3: 
        
        X_c1_w_opt = load_matrix('../data/cache/X_c1_w_opt.txt')

        # Convert the homogeneous 2D points to the required format for imagePoints
        imagePoints = np.ascontiguousarray(x3_data[0:2, :].T).reshape((x3_data.shape[1], 1, 2))

        # Estimate the pose of Camera 3 with respect to Camera 1
        # retval, theta_c3_c1_initial, t_c3_c1_initial = cv2.solvePnP(X_c1_w_opt, imagePoints, K_C, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)

        retval, theta_c3_c1_initial, t_c3_c1_initial = cv2.solvePnP(X_c1_w_opt, imagePoints, K_C, np.zeros((4, 1)), flags=cv2.SOLVEPNP_EPNP)

        R_c3_c1_initial = expm(crossMatrix(theta_c3_c1_initial))



        # Convert rotation vector to rotation matrix
        save_matrix('../data/cache/R_c3_c1_initial.txt', R_c3_c1_initial)
        save_matrix('../data/cache/t_c3_c1_initial.txt', t_c3_c1_initial)
        save_matrix('../data/cache/theta_c3_c1_initial.txt', theta_c3_c1_initial)

        print("Rotation matrix (Camera 3 with respect to Camera 1):\n", R_c3_c1_initial)
        print("Translation vector (Camera 3 with respect to Camera 1):\n", t_c3_c1_initial)
    
    if excercise4:
        R_c3_c1_initial = load_matrix('../data/cache/R_c3_c1_initial.txt')
        t_c3_c1_initial = load_matrix('../data/cache/t_c3_c1_initial.txt')
        theta_c3_c1_initial = load_matrix('../data/cache/theta_c3_c1_initial.txt')
        theta_c2_c1_initial = load_matrix('../data/cache/theta_c2_c1_initial.txt')
        t_c2_c1_initial = load_matrix('../data/cache/t_c2_c1_initial.txt')
        X_c1_w_opt = load_matrix('../data/cache/X_c1_w_opt.txt')
        X_c1_w_initial = load_matrix('../data/cache/X_c1_w_initial.txt')

        if excercise4_full:

            # Step 3: Set up the initial parameter vector for least_squares
            X_initial = X_c1_w_initial[:3, :].flatten()


            # TODO: BY-PASS 1
            initial_guess = np.hstack((theta_c2_c1_initial, t_c2_c1_initial, theta_c3_c1_initial, 
                                    t_c3_c1_initial, X_initial))
            
            
            
            # TODO lm model.
            # Step 4: Perform bundle adjustment with least_squares
            # nPoints = x1_data.shape[1]
            # result = least_squares(
            #     resBundleProjection3Views12DoF,
            #     initial_guess,
            #     args=(x1_data, x2_data, x3_data, K_C, nPoints),
            #     method='trf',
            #     jac='3-point',
            #     loss='huber',
            #     max_nfev=5000,
            #     verbose=2
            # )
            print("Start of least")
            nPoints = x1_data.shape[1]
            result = least_squares(
                resBundleProjection3Views12DoF,
                initial_guess,
                args=(x1_data, x2_data, x3_data, K_C, nPoints),
                method='lm'
                )
            print("Endo fo least")

            # Extract optimized parameters
            theta_c2_c1_opt = result.x[:3]
            t_c2_c1_opt = result.x[3:6]
            theta_c3_c1_opt = result.x[6:9]
            t_c3_c1_opt = result.x[9:12]
            X_c1_w_opt = result.x[12:].reshape(3,-1)

            # Save 
            save_matrix('../data/cache/theta_c2_c1_opt.txt', theta_c2_c1_opt)
            save_matrix('../data/cache/t_c2_c1_opt.txt', t_c2_c1_opt)
            save_matrix('../data/cache/theta_c3_c1_opt.txt', theta_c3_c1_opt)
            save_matrix('../data/cache/t_c3_c1_opt.txt', t_c3_c1_opt)
            save_matrix('../data/cache/X_c1_w_opt.txt', X_c1_w_opt)

        #Load the optimized values
        theta_c2_c1_opt = load_matrix('../data/cache/theta_c2_c1_opt.txt')
        t_c2_c1_opt = load_matrix('../data/cache/t_c2_c1_opt.txt')
        theta_c3_c1_opt = load_matrix('../data/cache/theta_c3_c1_opt.txt')
        t_c3_c1_opt = load_matrix('../data/cache/t_c3_c1_opt.txt')
        X_c1_w_opt = load_matrix('../data/cache/X_c1_w_opt.txt')


        X_c1_w_opt_h = np.vstack((X_c1_w_opt, np.ones((1, X_c1_w_opt.shape[1]))))

        # Convert rotation vectors back to matrices for visualization
        R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
        R_c3_c1_opt = expm(crossMatrix(theta_c3_c1_opt))

        # Print results
        print("Optimized Camera 2 Rotation Vector:", theta_c2_c1_opt)
        print("Optimized Camera 2 Translation Vector:", t_c2_c1_opt)
        print("Optimized Camera 3 Rotation Angle (around y-axis):", theta_c3_c1_opt)
        print("Optimized Camera 3 Translation (along z-axis):", t_c3_c1_opt)

        # Initiliaze the visualization
        R_c2_c1_initial = expm(crossMatrix(theta_c2_c1_initial))
        T_c2_c1_initial = ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
       
        T_c3_c1_initial = ensamble_T(R_c3_c1_initial,t_c3_c1_initial)
       
        T_c1_c2_initial = np.linalg.inv(T_c2_c1_initial)
        T_c1_c3_initial = np.linalg.inv(T_c3_c1_initial)

        R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
        T_c2_c1_opt = ensamble_T(R_c2_c1_opt,t_c2_c1_opt)

        R_c3_c1_opt = expm(crossMatrix(theta_c3_c1_opt))
        T_c3_c1_opt = ensamble_T(R_c3_c1_opt,t_c3_c1_opt)

        T_c1_c2_opt = np.linalg.inv(T_c2_c1_opt)
        T_c1_c3_opt = np.linalg.inv(T_c3_c1_opt)

        P_c1_c1 = get_projection_matrix(K_C, np.eye(4))
        P_c2_c1_opt = get_projection_matrix(K_C, T_c2_c1_opt)
        P_c3_c1_opt = get_projection_matrix(K_C, T_c3_c1_opt)

        x1_proj_opt = project_to_camera(P_c1_c1, X_c1_w_opt_h)
        x2_proj_opt = project_to_camera(P_c2_c1_opt, X_c1_w_opt_h)
        x3_proj_opt = project_to_camera(P_c3_c1_opt, X_c1_w_opt_h)

        # View residuals from optimized calculations
        fig, axs = plt.subplots(1, 3, figsize=(18, 6))
        visualize_residuals(image1, x1_data, x1_proj_opt, "Optimized Residuals in Image 1",ax=axs[0])
        visualize_residuals(image2, x2_data, x2_proj_opt, 'Optimized Residuals in Image 2',ax=axs[1])
        visualize_residuals(image3, x3_data, x3_proj_opt, 'Optimized Residuals in Image 3',ax=axs[2])
        plt.tight_layout()
        plt.show()

        
        # # Calculate the scale factor
        # scale_factor = get_scale_factor(X_w, X_c1_w_opt)
        # # Apply the scale factor to the optimized 3D points
        # X_c1_w_opt = X_c1_w_opt.T
        # X_c1_w_opt_scaled = X_c1_w_opt[:3, :] * scale_factor
        # X_c1_w_opt_scaled = np.vstack((X_c1_w_opt_scaled, np.ones((1, X_c1_w_opt.shape[1]))))  # Convert back to homogeneous coordinates
        # # Scale transformations
        # T_c2_c1_opt_scaled = T_c2_c1_opt * scale_factor
        # T_c3_c1_opt_scaled = T_c3_c1_opt * scale_factor

        # X_c1_w_opt = X_c1_w_opt.T
        visualize_3D_3cameras_optimization(T_w_c1, T_w_c2, T_w_c3, X_w, T_c1_c2_initial, 
                                           T_c1_c3_initial, T_c1_c2_opt, T_c1_c3_opt, X_c1_w_initial, X_c1_w_opt_h)


    

