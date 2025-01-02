import numpy as np
import matplotlib.pyplot as plt

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

    # # Connect the click event on image 1 to the handler
    # fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, F , ax1, ax2, img2,show_epipoles))
    # print('\nClose the figure to continue. Select a point from Img1 to get the equivalent epipolar line.')
    # plt.show(block=True)
    
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