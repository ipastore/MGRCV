import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.linalg import expm, logm, rq
from scipy.optimize import least_squares
import cv2
from random import sample

def compute_normalization_matrix(points):
    # Calculate mean for centering
    mean_x, mean_y = np.mean(points[0, :]), np.mean(points[1, :])
    std_dev = np.std(np.sqrt((points[0, :] - mean_x)**2 + (points[1, :] - mean_y)**2))

    # Construct transformation matrix
    T = np.array([
        [np.sqrt(2) / std_dev, 0, -mean_x * np.sqrt(2) / std_dev],
        [0, np.sqrt(2) / std_dev, -mean_y * np.sqrt(2) / std_dev],
        [0, 0, 1]
    ])
    return T

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

def eight_point_algorithm(x1, x2, image1, image2):
    """
        Compute the fundamental matrix using the eight-point algorithm.
        - Input:
            · x1, x2: Corresponding points from image 1 and image 2 (shape: 2 x N)
        -Output:
            · F: The estimated fundamental matrix
    """
    
    # Normalize the points
    T_1 = compute_normalization_matrix(x1)
    T_2 = compute_normalization_matrix(x2)
    x1_norm = T_1 @ x1
    x2_norm = T_2 @ x2

    
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
    F = T_2.T @ F_norm @ T_1
    return F


def calculate_epipolar_distance(points, lines):
    """
    Calculate the distance from each point in the first image to its corresponding epipolar line.
    - Input:
        points: Nx3 array of points in homogeneous coordinates from the first image (shape: (3, N) or (N, 3)).
        lines: Nx3 array of epipolar lines in homogeneous coordinates corresponding to the points (shape: (3, N) or (N, 3)).
    - Output:
        distances: An array of distances from each point to its corresponding line.
    """

    # Calculate the numerator (ax + by + c) for each point-line pair
    numerators = np.abs(lines[0] * points[0, :] + lines[1] * points[1, :] + lines[2]) 
    
    # Calculate the denominator (sqrt(a^2 + b^2)) for each line
    denominators = np.sqrt(lines[0]**2 + lines[1]**2)
    
    # Compute the distances
    distances = numerators / denominators
    return distances


def ransac_fundamental_matrix(matched_keypoints1, matched_keypoints2, image1, image2, nIter=1000, P = 3, threshold=0.01):
    """
    RANSAC algorithm to estimate the fundamental matrix between two images.
    - input:
        matched_keypoints1: Nx2 matrix of keypoints in image 1
        matched_keypoints2: Nx2 matrix of keypoints in image 2
        nIter: number of RANSAC iterations
        threshold: threshold for considering a point an inlier
    - output:
        best_F: best fundamental matrix estimated by RANSAC
        best_inliers: mask of inliers corresponding to best_F
    """
    

    print("Número de iteraciones: ", nIter)
    best_inliers_count = 0
    best_F = None
    rng = np.random.default_rng()

    for i in range(nIter):
        # Select 8 random matches
        idx = rng.choice(matched_keypoints1.shape[1], 8, replace=False)
        sample_points1 = matched_keypoints1[:,idx]
        sample_points2 = matched_keypoints2[:,idx]

        # Estimate F from the sampled points
        F = eight_point_algorithm(sample_points1, sample_points2, image1, image2)

        # Compute epipolar lines and count inliers
        lines_in_img2 = F @ matched_keypoints1
        
        errors_img2 = np.abs(calculate_epipolar_distance(matched_keypoints2, lines_in_img2))
        inliers = (errors_img2 < threshold)
        inliers_count = np.sum(inliers)

        # Update best F if more inliers found
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_F = F
            best_inliers = inliers
            
    print("Más votos conseguidos: ", best_inliers_count)
    if best_inliers is not None:
            inliers_points1 = matched_keypoints1[:, best_inliers]
            inliers_points2 = matched_keypoints2[:, best_inliers]
            best_F = eight_point_algorithm(inliers_points1, inliers_points2, image1, image2)

    return best_F, best_inliers

def visualize_epipolar_lines(F, img1, img2, show_epipoles=False, automatic=False):
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

    # # Connect the click event on image 1 to the handler
    if automatic:
            # Dimensiones de la imagen 1
        img1_height, img1_width = img1.shape[:2]
        
        # Seleccionar 5 puntos aleatorios
        random_points = np.column_stack((
            np.random.randint(0, img1_width, size=5),
            np.random.randint(0, img1_height, size=5)
        ))

        # Dibujar puntos y líneas epipolares
        for point in random_points:
            x, y = point
            ax1.plot(x, y, 'rx', markersize=10, label=f"Point ({x}, {y})")  # Marcar punto en img1
            ax1.annotate(f'({x}, {y})', (x, y), color='white', fontsize=8)
            
            # Convertir el punto a coordenadas homogéneas
            x1_homogeneous = np.array([x, y, 1])
            
            # Dibujar línea epipolar correspondiente en img2
            plot_epipolar_line(F, x1_homogeneous, ax2, img2, show_epipoles)

        if show_epipoles:
            # Calcular y graficar los epípolos
            u, s, vt = np.linalg.svd(F)
            epipole1 = vt[-1] / vt[-1][-1]  # Epíolo en la imagen 1
            epipole2 = u[:, -1] / u[-1, -1]  # Epíolo en la imagen 2
            ax1.plot(epipole1[0], epipole1[1], 'bo', label='Epipole in Img1')
            ax2.plot(epipole2[0], epipole2[1], 'bo', label='Epipole in Img2')

        ax2.legend()
        plt.show()
        plt.close(fig)
    else:
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, F , ax1, ax2, img2,show_epipoles))
        print('\nClose the figure to continue. Select a point from Img1 to get the equivalent epipolar line.')
        plt.show(block=True)
        plt.close(fig)
    
    
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
    # Gráfico para la primera pose
    # Crear una figura y un eje vacío en 3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.show()
    plt.close(fig)
    fig3D = draw_possible_poses(ax, R1_minust, X2)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_t')
    plt.show()
    plt.close()
    #P2_2
    fig3D = draw_possible_poses(ax, R1_minust, X2)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R1_minust')
    plt.show()
    plt.close()
    #P2_3
    fig3D = draw_possible_poses(ax, R2_t, X3)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_t')
    plt.show()
    plt.close()
    #P2_4
    fig3D = draw_possible_poses(ax, R2_minust, X4)
    adjust_plot_limits(ax, X1)
    plt.title('Possible Pose R2_minust')
    plt.show()
    plt.close()

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


def load_matrix(file_path):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")
    try:
        return np.loadtxt(file_path)
    except Exception as e:
        raise ValueError(f"Error loading matrix from {file_path}: {str(e)}")

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
    
def crossMatrixInv(M):
    """Extracts a vector x from a skew-symmetric matrix M."""
    x = [M[2, 1], M[0, 2], M[1, 0]]
    return np.array(x)

def resBundleProjection(Op, x1Data, x2Data, K_c, nPoints, verbose=False):
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

    if verbose:
        print("\n residuals 1:")
        print(res_x1_total)
        print("\n residuals 2:")
        print(res_x2_total)

    residuals = np.hstack((res_x1.flatten(), res_x2.flatten()))


    return residuals

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
            R = expm(crossMatrix(rotations[cam_idx - 2]))
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
        projected_2D = project_to_camera(P, points_h)  # (2, N)

        # Residuales (vectorizados)
        residuals_2D = keypoints_2D_visible - projected_2D[0:2,:]
        residuals.extend(residuals_2D.flatten())

    return np.array(residuals)


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

    initial_guess_theta_cam2 = crossMatrixInv(logm(R_cam2.astype('float64')))
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
        initial_guess_theta_cam = crossMatrixInv(logm(R_cam.astype('float64')))
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

    # # **Cámara de referencia** (fija, no está en el vector optimizado)
    # rotations.append(np.zeros(3))  # Rotación identidad (logm(R_ref) = [0, 0, 0])
    # translations.append(np.zeros(3))  # Traslación fija en el origen [0, 0, 0]

    # **Cámara 2 (en coordenadas polares)**
    theta_c2_ref = op_vector[offset:offset+3]
    t_theta = op_vector[offset+3]
    t_phi = op_vector[offset+4]
    t_c2_ref = np.array([
        np.sin(t_theta) * np.cos(t_phi),
        np.sin(t_theta) * np.sin(t_phi),
        np.cos(t_theta)
    ])
    R_new = expm(crossMatrix(theta_c2_ref))
    rotations.append(R_new)
    translations.append(t_c2_ref)
    offset += 5

    # **Cámaras adicionales (en cartesiano)**
    for _ in range(3, n_cameras + 1):  # Desde cam3 en adelante
        theta = op_vector[offset:offset+3]
        t = op_vector[offset+3:offset+6]
        R_new = expm(crossMatrix(theta))
        rotations.append(R_new)
        translations.append(t)
        offset += 6
    
    # **Puntos 3D**
    points_3D = op_vector[offset:].reshape(3, n_points).T

    return {
        "rotations": rotations,        # Rotaciones optimizadas (logm)
        "translations": translations,  # Traslaciones optimizadas
        "points_3D": points_3D         # Coordenadas optimizadas de puntos 3D
    }
    
    
def get_image(image_id):
    image_path = os.path.join(os.path.dirname(__file__), f'../Images/Set_12MP/EntireSet/{image_id}.jpg')
    return plt.imread(image_path) 

def get_matrix(image_id):
    image_path = os.path.join(os.path.dirname(__file__), f'../Images/Set_12MP/EntireSet/{image_id}.jpg')
    return plt.imread(image_path) 


# Function to load npz data
def load_npz_data(base_path, reference_image, target_image):
    npz_path = os.path.join(
        os.path.dirname(__file__), 
        f'{base_path}/{reference_image}_vs_{target_image}_inliers.npz'
    )
    if not os.path.exists(npz_path):
        raise FileNotFoundError(f"No se encontró el archivo: {npz_path}")
    return np.load(npz_path)



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
        R_pnp = expm(crossMatrix(rotation_vector))
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
        P_ref = get_projection_matrix(K_C, np.eye(4))
        T_cam3_to_ref = ensamble_T(R_pnp, t_pnp)
        P_cam3 = get_projection_matrix(K_C, T_cam3_to_ref)

        # Triangular nuevos puntos 3D
        new_points_3D = triangulate_points(x_ref_h, x_cam3_h, P_ref, P_cam3)
        new_points_3D = new_points_3D[:3, :].T.reshape(-1, 3)

        # Actualizar el diccionario con los nuevos puntos
        for i, match in enumerate(new_matches):
            new_point_id = len(point3D_map)  # Generar nuevo ID para el punto 3D
            point3D_map[new_point_id] = {ref_cam_id: match[0], target_cam_id: match[1]}
            optimized_3D_points = np.vstack((optimized_3D_points, new_points_3D[i,:]))

    return R_pnp, t_pnp, point3D_map, optimized_3D_points


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
    drawRefSystem(ax, np.eye(4), '-', 'C1')
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


def compute_projection_matrix_dlt(points_3d, points_2d):
    """
    Calcula la matriz de proyección P usando DLT (Direct Linear Transform).
    """
    num_points = points_3d.shape[0]
    A = np.zeros((2 * num_points, 12))
    for i in range(num_points):
        X, Y, Z, W = points_3d[i]
        x, y, w = points_2d[i]
        A[2 * i] = [X, Y, Z, W, 0, 0, 0, 0, -x * X, -x * Y, -x * Z, -x * W]
        A[2 * i + 1] = [0, 0, 0, 0, X, Y, Z, W, -y * X, -y * Y, -y * Z, -y * W]
    U, S, Vt = np.linalg.svd(A)
    P = Vt[-1].reshape((3, 4))
    P /= P[2, 3]
    return P

def ransac_dlt(points_3d, points_2d, threshold=5, max_iterations=1000):
    """
    Aplica RANSAC para calcular la matriz de proyección P eliminando outliers.

    Parámetros:
    - points_3d: numpy array de forma (n, 4), puntos en coordenadas homogéneas.
    - points_2d: numpy array de forma (n, 3), puntos en coordenadas homogéneas.
    - threshold: umbral para clasificar inliers (en píxeles).
    - max_iterations: número máximo de iteraciones.

    Retorna:
    - best_P: matriz de proyección P optimizada.
    - inliers: boolean array indicando las correspondencias consideradas inliers.
    """
    num_points = points_3d.shape[0]
    best_P = None
    best_inliers = []

    for _ in range(max_iterations):
        # Selección aleatoria de 6 correspondencias
        indices = sample(range(num_points), 6)
        subset_3d = points_3d[indices]
        subset_2d = points_2d[indices]

        # Calcular matriz P provisional
        P = compute_projection_matrix_dlt(subset_3d, subset_2d)

        # Calcular reproyección para todas las correspondencias
        projected = (P @ points_3d.T).T
        projected /= projected[:, 2:3]  # Normalizar coordenadas homogéneas
        errors = np.linalg.norm(projected[:, :2] - points_2d[:, :2], axis=1)

        # Determinar inliers
        inliers = errors < threshold

        # Guardar el mejor modelo
        if sum(inliers) > len(best_inliers):
            best_P = P
            best_inliers = inliers

    return best_P, best_inliers



def decompose_projection_matrix_with_sign(P):
    """
    Descompone la matriz de proyección P usando el signo del determinante de M.
    """
    # Extraer M (los primeros 3x3 de P)
    M = P[:, :3]

    # Calcular el signo del determinante de M
    sign = np.sign(np.linalg.det(M))

    # Ajustar P con el signo
    P_adjusted = sign * P

    # Llamar a la función de OpenCV
    K, R_cw, t_wc, _, _, _, _ = cv2.decomposeProjectionMatrix(P_adjusted)

    # Normalizar t_wc para que esté en coordenadas homogéneas
    t_wc /= t_wc[-1]

    return K, R_cw, t_wc


def reprojection_error_cartesian(params, points_3d, points_2d, K):
    """
    Calcula el error de reproyección dado R (en representación logarítmica)
    y t (en coordenadas cartesianas).

    Parámetros:
    - params: Vector con [theta_x, theta_y, theta_z, t_x, t_y, t_z].
    - points_3d: Puntos 3D (n, 4).
    - points_2d: Puntos 2D observados (n, 3).
    - K: Matriz intrínseca de la cámara (3x3).

    Retorna:
    - error: Vector de reproyección para cada punto.
    """
    # Extraer parámetros
    theta = params[:3]  # Rotación en logm (vector cartesiano)
    t = params[3:].reshape((3, 1))  # Traslación en coordenadas cartesianas

    # Reconstruir la matriz de rotación R usando expm
    R = expm(crossMatrix(theta))
    
    # Proyección de los puntos 3D
    P = K @ np.hstack((R, t))  # Matriz de proyección: P = K [R|t]
    projected = (P @ points_3d.T).T  # Proyección de los puntos 3D en 2D
    projected /= projected[:, 2:3]  # Normalizar coordenadas homogéneas

    # Calcular error de reproyección
    error = projected[:, :2] - points_2d[:, :2]
    return error.flatten()

def bundle_adjustment_old(points_3d, points_2d, K, R_init, t_init):
    """
    Optimiza R y t para minimizar el error de reproyección, usando representación logarítmica
    para R y coordenadas cartesianas para t.

    Parámetros:
    - points_3d: Puntos 3D (n, 4).
    - points_2d: Puntos 2D observados (n, 3).
    - K: Matriz intrínseca de la cámara (3x3).
    - R_init: Matriz inicial de rotación (3x3).
    - t_init: Vector inicial de traslación (3x1).

    Retorna:
    - R_opt: Matriz optimizada de rotación (3x3).
    - t_opt: Vector optimizado de traslación (3x1).
    """
    # Convertir R_init a su representación logarítmica
    theta_init = crossMatrixInv(logm(R_init))

    # Crear el vector inicial de parámetros (theta + t)
    params_init = np.hstack((theta_init, t_init.flatten()))

    # Optimización con Levenberg-Marquardt
    result = least_squares(
        reprojection_error_cartesian,
        params_init,
        args=(points_3d, points_2d, K),
        method='lm'
    )

    # Extraer parámetros optimizados
    theta_opt = result.x[:3]  # Rotación optimizada en logm
    t_opt = result.x[3:].reshape((3, 1))  # Traslación optimizada

    # Reconstruir R optimizada
    R_opt = expm(crossMatrix(theta_opt))
    
    return R_opt, t_opt


def compute_rmse(observed_2D, projected_2D):
    """
    Calcula el RMSE entre los puntos observados y proyectados.

    Args:
        observed_2D (np.ndarray): Puntos observados (2xN).
        projected_2D (np.ndarray): Puntos proyectados (2xN).
        
    Returns:
        float: RMSE entre los puntos observados y proyectados.
    """
    squared_errors = np.linalg.norm(observed_2D - projected_2D, axis=0) ** 2
    return np.sqrt(np.mean(squared_errors))


def load_points_from_colmap(file_path):
    points = []
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith('#') or line.strip() == '':
                continue
            parts = line.strip().split()
            x, y, z = map(float, parts[1:4])  # Coordenadas 3D
            points.append([x, y, z])
    return np.array(points)



from scipy.spatial.transform import Rotation as R


def extract_camera_poses(images_file):
    """
    Extract camera poses from a COLMAP-style images.txt file.
    
    -input:
        images_file: str
            Path to the images.txt file containing the camera poses.

    -output:
        camera_poses: dict
            Dictionary mapping image names to their corresponding poses.
            Each pose contains:
                - image_id: int
                    Unique identifier of the image.
                - rotation_matrix: ndarray (3x3)
                    Rotation matrix representing the camera orientation.
                - translation_vector: ndarray (3,)
                    Translation vector representing the camera position.
    """
    camera_poses = {}
    
    with open(images_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if line.startswith("#") or len(line.strip()) == 0:
            # Skip comments and empty lines
            continue
        
        parts = line.strip().split()
        if len(parts) != 10 :
            # Skip the second line of the entry (2D-3D correspondences)
            continue
        
        # Extract pose data
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]
        
        # Compute rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        # rotation_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # R_mat = rotation_mat.as_matrix()
        translation_vector = np.array([tx, ty, tz])

        # Store pose
        camera_poses[image_name] = {
            "image_id": image_id,
            "rotation_matrix": rotation_matrix,
            "translation_vector": translation_vector
        }
    
    return camera_poses

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert quaternion to rotation matrix.
    
    -input:
        qw: float
            Scalar component of the quaternion.
        qx, qy, qz: float
            Vector components of the quaternion.

    -output:
        r: ndarray (3x3)
            Rotation matrix corresponding to the input quaternion.
    """
    r = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return r


def get_camera_axes_colmap(position, rotation, scale=0.1):
    """
    Calcula los ejes X, Y, Z de la cámara en el espacio 3D.
    
    Args:
        position (np.ndarray): Posición de la cámara (3,).
        rotation (np.ndarray): Matriz de rotación de la cámara (3x3).
        scale (float): Escala para visualizar los ejes.
    
    Returns:
        dict: Diccionario con los ejes X, Y, Z como vectores 3D.
    """
    x_axis = position + scale * rotation[:, 0]  # Eje X
    y_axis = position + scale * rotation[:, 1]  # Eje Y
    z_axis = position + scale * rotation[:, 2]  # Eje Z (dirección de la cámara)
    return {"x": x_axis, "y": y_axis, "z": z_axis}

def changePoseTransformation(R_wc, t_wc):

    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    return R_cw, t_cw   


def save_transformations(R_list, t_list, rotation_file="rotations.npz", translation_file="translations.npz"):
    """
    Guarda las matrices de rotación y los vectores de traslación en archivos separados.

    Args:
        R_list (list of np.ndarray): Lista de matrices de rotación (3x3 cada una).
        t_list (list of np.ndarray): Lista de vectores de traslación (3x1 cada uno).
        rotation_file (str): Nombre del archivo para guardar las rotaciones.
        translation_file (str): Nombre del archivo para guardar las traslaciones.
    """
    output_dir = os.path.join(os.path.dirname(__file__), 'results')
    os.makedirs(output_dir, exist_ok=True)
    # Guardar matrices de rotación
    np.savez(os.path.join(output_dir, rotation_file), *R_list)
    print(f"Rotaciones guardadas en {rotation_file}")

    # Guardar vectores de traslación
    np.savez(os.path.join(output_dir, translation_file), *t_list)
    print(f"Traslaciones guardadas en {translation_file}")
    
    

def load_transformations(rotation_file="rotations.npz", translation_file="translations.npz"):
    """
    Carga las matrices de rotación y los vectores de traslación desde archivos separados.

    Args:
        rotation_file (str): Nombre del archivo de las rotaciones.
        translation_file (str): Nombre del archivo de las traslaciones.

    Returns:
        tuple: (R_list, t_list)
            R_list (list of np.ndarray): Lista de matrices de rotación (3x3 cada una).
            t_list (list of np.ndarray): Lista de vectores de traslación (3x1 cada uno).
    """
    # Cargar matrices de rotación
    with np.load(rotation_file) as data:
        R_list = [data[key] for key in data.files]
    print(f"Rotaciones cargadas desde {rotation_file}")

    # Cargar vectores de traslación
    with np.load(translation_file) as data:
        t_list = [data[key] for key in data.files]
    print(f"Traslaciones cargadas desde {translation_file}")

    return R_list, t_list
