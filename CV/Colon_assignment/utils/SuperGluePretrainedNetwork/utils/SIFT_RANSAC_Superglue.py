#####################################################################################
#
# MRGCV Unizar - Computer vision - Laboratory 2
#
# Title: SIFT matching
#
# Date: 28 September 2020
#
#####################################################################################
#
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.0
#
#####################################################################################

import matplotlib.pyplot as plt
import numpy as np
import cv2
import random


def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def indexMatrixToMatchesList(matchesList):
    """
     -input:
         matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     -output:
        dMatchesList: list of n DMatch object
     """
    dMatchesList = []
    for row in matchesList:
        dMatchesList.append(cv2.DMatch(_queryIdx=row[0], _trainIdx=row[1], _distance=row[2]))
    return dMatchesList

def matchesListToIndexMatrix(dMatchesList):
    """
     -input:
         dMatchesList: list of n DMatch object
     -output:
        matchesList: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
     """
    matchesList = []
    for k in range(len(dMatchesList)):
        matchesList.append([int(dMatchesList[k].queryIdx), int(dMatchesList[k].trainIdx), dMatchesList[k].distance])
    return matchesList

def matchWith2NNDR(desc1, desc2, distRatio, minDist):
    """
    Nearest Neighbours Matching algorithm checking the Distance Ratio.
    A match is accepted only if its distance is less than distRatio times
    the distance to the second match.
    -input:
        desc1: descriptors from image 1 nDesc x 128
        desc2: descriptors from image 2 nDesc x 128
        distRatio:
    -output:
       matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.sqrt(np.sum((desc2 - desc1[kDesc1, :]) ** 2, axis=1))
        indexSort = np.argsort(dist)
        if (dist[indexSort[0]] < minDist):
            matches.append([kDesc1, indexSort[0], dist[indexSort[0]]])
    return matches

def matchWith2NNDR_v2(desc1, desc2, distRatio, minDist):
    """
        Nearest Neighbours Matching algorithm checking the Distance Ratio.
        A match is accepted only if its distance is less than distRatio times
        the distance to the second match.
        -input:
            desc1: descriptors from image 1 nDesc x 128
            desc2: descriptors from image 2 nDesc x 128
            distRatio:
        -output:
            matches: nMatches x 3 --> [[indexDesc1,indexDesc2,descriptorDistance],...]]
    """
    matches = []
    nDesc1 = desc1.shape[0]
    for kDesc1 in range(nDesc1):
        dist = np.linalg.norm(desc2 - desc1[kDesc1, :], axis=1)
        indexSort = np.argsort(dist)
        d1 = dist[indexSort[0]]  # Smallest distance (nearest neighbor)
        d2 = dist[indexSort[1]]  # Second smallest distance (second nearest neighbor)
        if (d1<d2*distRatio ) and (d1 < minDist):
            matches.append([kDesc1, indexSort[0], d1])
    return matches

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

def draw_matches_with_hypothesis(image1, kp1, image2, kp2, matches, hypothesis_matches=None, inliers=None, title="Matches"):
    """
    Draw matches between two images
    -input:
        image1: image 1
        kp1: keypoints from image 1
        image2: image 2
        kp2: keypoints from image 2
        matches: list of DMatch objects
        hypothesis_matches: list of DMatch objects with the hypothesis
        inliers: list of DMatch objects with the inliers
        title: title of the plot
    """
    img_matches = cv2.drawMatches(image1, kp1, image2, kp2, matches, None,
                                  matchColor=(255, 0, 0),
                                  flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if hypothesis_matches:
        img_matches = cv2.drawMatches(image1, kp1, image2, kp2, hypothesis_matches, img_matches,
                                      matchColor=(0, 255, 0),
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    if inliers:
        img_matches = cv2.drawMatches(image1, kp1, image2, kp2, inliers, img_matches,
                                      matchColor=(0, 0, 255),
                                      flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    plt.figure()
    plt.imshow(img_matches, cmap='gray')
    plt.title(title)
    plt.show()

def ransac_homography(matched_keypoints1, matched_keypoints2, nIter = 1000, RANSACThreshold=5.0):
    """
    RANSAC algorithm to estimate the homography between two images
    -input:
        keypoints1: keypoints from image 1
        keypoints2: keypoints from image 2
        matches: list of DMatch objects
        nIter: number of iterations
        RANSACThreshold: tolerance to accept a point as inlier
    """
    n_inliers_max = 0               # Number of inliers 
    best_H = None                   # Best homography matrix 
    best_inliers_mask = None        # Mask of inliers
    rng = np.random.default_rng()   # Random number generator
    plot_interval = 500             # Plot interval
    
    # Ensure points are in homogeneous coordinates (3xN)
    N = matched_keypoints1.shape[1]
    if matched_keypoints1.shape[0] == 2:
        matched_keypoints1 = np.vstack((matched_keypoints1, np.ones((1, N))))
    if matched_keypoints2.shape[0] == 2:
        matched_keypoints2 = np.vstack((matched_keypoints2, np.ones((1, N))))
        
    for kAttempt in range(nIter):
        # Take four random matched points (minimal set defining our model)
        idx = rng.choice(matched_keypoints1.shape[1], 4, replace=False)
        points1 = matched_keypoints1[:,idx]
        points2 = matched_keypoints2[:,idx]
        
        # Estimate the homography matrix
        H = estimate_homography_from_points(points1, points2)
        
        # We must tranform all points in image 1 to image 2 using the estimated homography
        points1_transformed = H @ matched_keypoints1
        points1_transformed /= points1_transformed[2, :]
        
        # We must calculate the error for each point (L2 distance)
        errors = np.linalg.norm(points1_transformed[:2,:] - matched_keypoints2[:2,:], axis=0)
        
        # Count inliers
        inliers = errors < RANSACThreshold
        n_inliers = np.sum(inliers)
        
        # Update the best homography if more inliers are found
        if n_inliers > n_inliers_max:
            n_inliers_max = n_inliers
            best_H = H
            best_inliers_mask = inliers
            
    if best_inliers_mask is not None:
            inliers_points1 = matched_keypoints1[:, best_inliers_mask]
            inliers_points2 = matched_keypoints2[:, best_inliers_mask]
            best_H = estimate_homography_from_points(inliers_points1, inliers_points2)

    return best_H, best_inliers_mask

def drawLine(l,strFormat,lWidth):
    """
    Draw a line
    -input:
      l: image line in homogenous coordinates
      strFormat: line format
      lWidth: line width
    -output: None
    """
    # p_l_y is the intersection of the line with the axis Y (x=0)
    p_l_y = np.array([0, -l[2] / l[1]])
    # p_l_x is the intersection point of the line with the axis X (y=0)
    p_l_x = np.array([-l[2] / l[0], 0])
    # Draw the line segment p_l_x to  p_l_y
    plt.plot([p_l_y[0], p_l_x[0]], [p_l_y[1], p_l_x[1]], strFormat, linewidth=lWidth)

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


def eight_point_algorithm(x1, x2):
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


def ransac_fundamental_matrix(matched_keypoints1, matched_keypoints2, p=0.5, threshold=4, confidence=0.9999):
    """
    RANSAC algorithm to estimate the fundamental matrix between two images.
    - input:
        matched_keypoints1: Nx2 matrix of keypoints in image 1
        matched_keypoints2: Nx2 matrix of keypoints in image 2
        threshold: threshold for considering a point an inlier
        confidence: desired confidence level (default is 0.99)
    - output:
        best_F: best fundamental matrix estimated by RANSAC
        best_inliers: mask of inliers corresponding to best_F
    """
    
    best_inliers_count = 0
    best_F = None
    rng = np.random.default_rng()
    best_inliers = None

    # Calculate the initial number of iterations based on the confidence level
    s = 8     # Number of points needed to estimate the model
    nIter = int(np.log(1 - confidence) / np.log(1 - p**s))

    for i in range(nIter):
        # Select 8 random matches
        idx = rng.choice(matched_keypoints1.shape[1], 8, replace=False)
        sample_points1 = matched_keypoints1[:,idx]
        sample_points2 = matched_keypoints2[:,idx]

        # Estimate F from the sampled points
        F = eight_point_algorithm(sample_points1, sample_points2)

        # Compute epipolar lines and count inliers
        lines_in_img2 = F @ matched_keypoints1

        # Compute epipolar lines and count inliers
        errors_img2 = np.abs(calculate_epipolar_distance(matched_keypoints2, lines_in_img2))
        inliers = (errors_img2 < threshold)
        inliers_count = np.sum(inliers)

        # Update best F if more inliers found
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_F = F
            best_inliers = inliers

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

    # Connect the click event on image 1 to the handler
    fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, F , ax1, ax2, img2,show_epipoles))
    print('\nClose the figure to continue. Select a point from Img1 to get the equivalent epipolar line.')
    plt.show(block=True)

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


