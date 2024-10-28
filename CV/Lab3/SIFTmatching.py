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
        if (d1>d2*distRatio ) and (d1 < minDist):
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

def ransac_homography(matched_keypoints1, matched_keypoints2, matches, nIter = 1000, RANSACThreshold=5.0):
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

def ransac_fundamental_matrix(matched_keypoints1, matched_keypoints2, nIter=1000, threshold=0.01):
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
    best_inliers_count = 0
    best_F = None
    rng = np.random.default_rng()

    for i in range(nIter):
        # Select 8 random matches
        idx = rng.choice(len(matched_keypoints1), 8, replace=False)
        sample_points1 = matched_keypoints1[idx]
        sample_points2 = matched_keypoints2[idx]

        # Estimate F from the sampled points
        F = eight_point_algorithm(sample_points1, sample_points2)

        # Compute epipolar lines and count inliers
        lines1 = F @ matched_keypoints2.T
        lines2 = F.T @ matched_keypoints1.T
        errors1 = np.abs(np.sum(lines1 * matched_keypoints1.T, axis=0)) / np.sqrt(lines1[0]**2 + lines1[1]**2)
        errors2 = np.abs(np.sum(lines2 * matched_keypoints2.T, axis=0)) / np.sqrt(lines2[0]**2 + lines2[1]**2)
        inliers = (errors1 < threshold) & (errors2 < threshold)
        inliers_count = np.sum(inliers)

        # Update best F if more inliers found
        if inliers_count > best_inliers_count:
            best_inliers_count = inliers_count
            best_F = F
            best_inliers = inliers

    return best_F, best_inliers

if __name__ == '__main__':
    np.set_printoptions(precision=4, linewidth=1024, suppress=True)

    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = './data/image1.png'
    path_image_2 = './data/image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    distRatio = 0.8
    minDist = 200
    # matchesList = matchWith2NNDR(descriptors_1, descriptors_2, distRatio, minDist)
    matchesList = matchWith2NNDR_v2(descriptors_1, descriptors_2, distRatio, minDist)
    print(f"Number of matches for minDist {minDist}: {len(matchesList)}")
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)

    # Plot the first 10 matches
    imgMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
    plt.show(block=True)  # This will block execution until the figure is closed

    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)

    # Matching with minDist = 200 for RANSAC
    minDist_ransac = 700
    matchesList_ransac = matchWith2NNDR_v2(descriptors_1, descriptors_2, distRatio, minDist_ransac)
    print(f"Number of matches for minDist {minDist_ransac}: {len(matchesList_ransac)}")
    dMatchesList_ransac = indexMatrixToMatchesList(matchesList_ransac)
    dMatchesList_ransac = sorted(dMatchesList_ransac, key=lambda x: x.distance)

    # Convert matches and prepare for RANSAC with minDist = 200
    matchesList_ransac = matchesListToIndexMatrix(dMatchesList_ransac)

    # Matched points in numpy from list of DMatches for RANSAC
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList_ransac]).reshape(len(dMatchesList_ransac), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList_ransac]).reshape(len(dMatchesList_ransac), 2)

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))
    
    # RANSAC homography estimation
    H, inliers_mask = ransac_homography(x1, x2, dMatchesList_ransac, nIter=5000, RANSACThreshold=2.5)
    print("Homography Matrix H:\n", H)
    
    imgMatched_RANSAC = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2, dMatchesList[:100],
                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    inliers_matches = [dMatchesList[i] for i in range(len(dMatchesList)) if inliers_mask[i]]
    imgInliersMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2,
                                        inliers_matches, None, matchColor=(0, 255, 0),
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    outliers_matches = [dMatchesList[i] for i in range(len(dMatchesList)) if not inliers_mask[i]]
    imgOutliersMatched = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2,
                                         outliers_matches, None, matchColor=(0, 0, 255),
                                         flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)
    
    fig, (ax1, ax2) = plt.subplots(1, 2)
    # Configure the first subplot (Image 1)
    ax1.imshow(imgInliersMatched, cmap='gray')
    ax1.set_title("Inliers Matches")
    # Configure the second subplot (Image 2)
    ax2.imshow(imgOutliersMatched, cmap='gray')
    ax2.set_title("Outliers Matches")
    # Show the figure and wait until it is closed
    plt.show(block=True)  # This will block execution until the figure is closed

    
    ### COMPARING WITH SUPERGLUE ###
    # Load SuperGlue matches (substitute this with your SuperGlue inference code if needed)
    superglue_matches = np.load("./data/image1_image2_matches.npz")
    keypoints0 = superglue_matches["keypoints0"]  # Keypoints in image 1
    keypoints1 = superglue_matches["keypoints1"]  # Keypoints in image 2
    matches = superglue_matches["matches"]  # Indices of matches
    confidence = superglue_matches["match_confidence"]
    
    # Filter valid matches
    valid_matches = matches > -1
    matched_keypoints0 = keypoints0[valid_matches]
    matched_keypoints1 = keypoints1[matches[valid_matches]]

    # Convert SuperGlue matches to format compatible with RANSAC
    x1_superglue = np.vstack((matched_keypoints0.T, np.ones((1, matched_keypoints0.shape[0]))))
    x2_superglue = np.vstack((matched_keypoints1.T, np.ones((1, matched_keypoints1.shape[0]))))
    
    # Run RANSAC with SuperGlue matches
    H_superglue, inliers_mask_superglue = ransac_homography(x1_superglue, x2_superglue, valid_matches, 
                                                                   nIter=5000, RANSACThreshold=5.0,)

    # Compare with SIFT-based RANSAC result
    # H_sift and inliers_mask_sift were previously computed with SIFT-based RANSAC
    print("SIFT Homography Matrix H:\n", H)
    print("SuperGlue Homography Matrix H:\n", H_superglue)

    # Display results
    print(f"SIFT-based inliers: {np.sum(inliers_mask)}")
    print(f"SuperGlue-based inliers: {np.sum(inliers_mask_superglue)}")
    
    # Generate inliers and outliers from valid matches for SuperGlue
    inliers_matches_superglue = [
        cv2.DMatch(_queryIdx=i, _trainIdx=matches[i], _distance=0)
        for i in range(len(matches)) if valid_matches[i] and i < len(inliers_mask_superglue) and inliers_mask_superglue[i]
    ]

    outliers_matches_superglue = [
        cv2.DMatch(_queryIdx=i, _trainIdx=matches[i], _distance=0)
        for i in range(len(matches)) if valid_matches[i] and i < len(inliers_mask_superglue) and not inliers_mask_superglue[i]
    ]

    # Draw SuperGlue inliers and outliers
    imgInliersMatched_SG = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2,
                                        inliers_matches_superglue, None, matchColor=(0, 255, 0),
                                        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    imgOutliersMatched_SG = cv2.drawMatches(image_pers_1, keypoints_sift_1, image_pers_2, keypoints_sift_2,
                                            outliers_matches_superglue, None, matchColor=(0, 0, 255),
                                            flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS)

    # Display the comparison between SIFT and SuperGlue matches
    fig, ((ax1, ax2), (ax3, ax4), (ax5, ax6),) = plt.subplots(3, 2, figsize=(12, 8))
    ax1.imshow(imgMatched_RANSAC, cmap='gray')
    ax1.set_title("SIFT Matches")
    ax2.imshow(imgMatched_RANSAC, cmap='gray')
    ax2.set_title("SuperGlue Matches")
    ax3.imshow(imgInliersMatched, cmap='gray')
    ax3.set_title("Inliers Matches SIFT")
    ax4.imshow(imgOutliersMatched, cmap='gray')
    ax4.set_title("Outliers Matches SIFT")
    ax5.imshow(imgInliersMatched_SG, cmap='gray')
    ax5.set_title("Inliers Matches SuperGlue")
    ax6.imshow(imgOutliersMatched_SG, cmap='gray')
    ax6.set_title("Outliers Matches SuperGlue")
    plt.tight_layout()
    plt.show(block=True)  # Block execution until the figure is closed
        
        
    