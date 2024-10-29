import matplotlib.pyplot as plt
import numpy as np
import random
import scipy.linalg as scAlg
import cv2
from PIL import Image


def fundamental_matches(X1, X2):
    '''
    Computes the fundamental matrix H that relates two sets of corresponding points in images 1 and 2.
    
    X1 : 2D array containing the homogeneous coordinates of points in the first image [3, N].
    X2 : 2D array containing the homogeneous coordinates of corresponding points in the second image [3, N].
    '''
    Correspondences = []
    for i in range (X1.shape[1]): 
        x1 = X1[:, i].reshape(3, 1) # Vector [3,1]
        x2 = X2[:, i].reshape(1, 3) # Vector [1,3]

        Aux = x1 @ x2 # Auxiliar matrix to obtain the matrix multiplication between a match

        vector = Aux.flatten(order='F')  # Flatten matrix to vector using Fortran order (column by column)
        
        Correspondences.append(vector)
    

    Correspondences= np.array(Correspondences) 

    U, S, V = np.linalg.svd(Correspondences) # Decomposition SVD to obtain solution from last row of matrix V
    v4 = V[-1] 

    F = v4.reshape(3,3)
    
    return F


def epipole_epipolarLines_click_plot (image1, F, image2):
    """
    Plots epipolar lines on two images based on user-clicked points.
    - User clicks 5 points on 'image1'.
    - Corresponding epipolar lines are calculated using the fundamental matrix 'F'.
    - The epipole is determined via SVD of F.
    - Epipolar lines and the epipole are plotted on 'image2'.
    """
    
    F_test = np.loadtxt('F_21_test.txt')
    
    figure_1_id = 1
    plt.figure(figure_1_id)
    plt.imshow(image1)
    plt.title('Image 1 - Click a point (5 in total)')

    coord_clicked_points = []
    epipole_lines = []

    # Loop for all points
    for i in range(5):

        coord_clicked_point = plt.ginput(1, show_clicks=False)
        
        # Extract coordinates from clicked points and add homogeneous coordinate
        p_clicked = np.append(coord_clicked_point[0], 1)
        coord_clicked_points.append(p_clicked)
        
        ''' Epipolar line formula --> l = F @ x '''
        ep_line = F @ p_clicked 
        epipole_lines.append(ep_line)
        
        # Mark clicked point
        plt.plot(coord_clicked_point[0][0], coord_clicked_point[0][1], marker='x', color='red')
        plt.draw()


    epipole_lines = np.array(epipole_lines)
    
    ''' Epipole determination --> F @ e = 0 --> SVD '''
    U, S, V_true = np.linalg.svd(F_test.T)
    U, S, V = np.linalg.svd(F.T)
    epipole_true = V_true[-1] # Solution is last row of V from decomposition
    epipole = V[-1] 
    
    comp_hom_true = epipole_true[-1]
    epipole_true = epipole_true/ comp_hom_true

    comp_hom = epipole[-1]
    epipole = epipole/ comp_hom # Scaled epiople


    # Plot epipolar lines and epipole
    figure_2_id = 2
    plt.figure(figure_2_id)

    plt.imshow(image2, zorder=0) 

    plt.plot(epipole_true[0], epipole_true[1], marker='+', color='green')
    plt.plot(epipole[0], epipole[1], marker='+', color='blue')

    for i in range (5):
        
        drawLine(epipole_lines[i], 'r-', 2)  # 'r-' es la línea roja
        
    plt.show()

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

def matchWith2NDRR(desc1, desc2, distRatio, minDist):
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

def launchNndrSift():
    """
    Launch the NNDR SIFT algorithm
    -input: None
    -output: None
    """
    # Images path
    timestamp1 = '1403715282262142976'
    timestamp2 = '1403715413262142976'

    path_image_1 = 'image1.png'
    path_image_2 = 'image2.png'

    # Read images
    image_pers_1 = cv2.imread(path_image_1)
    image_pers_2 = cv2.imread(path_image_2)

    # Feature extraction
    sift = cv2.SIFT_create(nfeatures=0, nOctaveLayers = 5, contrastThreshold = 0.02, edgeThreshold = 20, sigma = 0.5)
    keypoints_sift_1, descriptors_1 = sift.detectAndCompute(image_pers_1, None)
    keypoints_sift_2, descriptors_2 = sift.detectAndCompute(image_pers_2, None)

    distRatio = 0.8
    minDist = 500
    matchesList = matchWith2NDRR(descriptors_1, descriptors_2, distRatio, minDist)
    dMatchesList = indexMatrixToMatchesList(matchesList)
    dMatchesList = sorted(dMatchesList, key=lambda x: x.distance)
    
    # Conversion from DMatches to Python list
    matchesList = matchesListToIndexMatrix(dMatchesList)

    # Matched points in numpy from list of DMatches
    srcPts = np.float32([keypoints_sift_1[m.queryIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    dstPts = np.float32([keypoints_sift_2[m.trainIdx].pt for m in dMatchesList]).reshape(len(dMatchesList), 2)
    
    matches_sift = np.empty([len(matchesList)], dtype=object)
    #print(srcPts[0])
    for i in range(len(matchesList)):
        matches_sift[i] = [srcPts[i], dstPts[i]] 

    # Matched points in homogeneous coordinates
    x1 = np.vstack((srcPts.T, np.ones((1, srcPts.shape[0]))))
    x2 = np.vstack((dstPts.T, np.ones((1, dstPts.shape[0]))))
    
    return matches_sift, keypoints_sift_1, keypoints_sift_2

def norm_F_pix (image1, image2, x1, x2):
    width_1, height_1 = image1.shape[:2]
    width_2, height_2 = image2.shape[:2]

    T_1 = np.array([
        [1/width_1, 0, -1/2],
        [0, 1/height_1, -1/2],
        [0, 0, 1]
        ])
    T_2 = np.array([
        [1/width_2, 0, -1/2],
        [0, 1/height_2, -1/2],
        [0, 0, 1]
        ])
    
    x1_norm = T_1 @ x1   
    x2_norm = T_2 @ x2
    
    F_norm_1 = fundamental_matches(x1_norm, x2_norm)
    
    U, S, V = np.linalg.svd(F_norm_1)
            
    S[2]=0
            
    F_norm = U @ np.diag(S) @ V

    F = T_2.T @ F_norm @ T_1
    
    
    return F
    
def ransac_F (matches, image1, image2, threshold, P, inlier_ratio):
    
    nAttemps= np.log(1 - P) / np.log(1 - inlier_ratio**8)
    nAttemps = int(nAttemps) 
    
    best_H = None
    best_votes = -1
    best_matches = None
    
    x1_all = [match[0] for match in matches]
    x2_all = [match[1] for match in matches]
    x1_all = np.vstack([np.asarray(x1_all).T, np.ones((1, len(x1_all)))])
    x2_all = np.vstack([np.asarray(x2_all).T, np.ones((1, len(x2_all)))])   
    x1_all = np.asarray(x1_all)
    x2_all = np.asarray(x2_all)
    
    for i in range(nAttemps):
        random_8 = np.random.randint(0, len(matches), 8)
        matches_8 = [matches[random_number] for random_number in random_8]
        x1 = [match[0] for match in matches_8]
        x2 = [match[1] for match in matches_8]
        
        x1 = np.asarray(x1).T
        x2 = np.asarray(x2).T

        x1 = np.vstack([x1, np.ones((1, x1.shape[1]))])
        x2 = np.vstack([x2, np.ones((1, x2.shape[1]))])
  
        F = norm_F_pix(image1, image2, x1, x2)
        
        epipolar_lines_x2_est = F @ x1_all
           
                
        error = np.abs(epipolar_lines_x2_est[0] * x2_all[0, :] +
                        epipolar_lines_x2_est[1] * x2_all[1, :] +
                        epipolar_lines_x2_est[2]) / np.sqrt(epipolar_lines_x2_est[0]**2 + epipolar_lines_x2_est[1]**2)
                  
        inliers = 0

        inliers = np.sum(error < threshold)
        inliers_indices = np.where(error < threshold)[0]

        if (inliers > best_votes):
                
            best_votes = inliers
            best_F = F
            
            best_matches = [matches[i] for i in random_8]
            best_inliers = [matches[i] for i in inliers_indices]
                
            best_x1 = x1
            best_x2 = x2
            
    print("Más votos conseguidos: ", best_votes)
        
    return best_F, best_matches, best_inliers
    
        
        
if __name__ == '__main__':
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)  
    # load the images
    image_1 = cv2.imread('image1.png')
    image_2 = cv2.imread('image2.png')
    # import the pairs .npz file
    matches_dict = np.load('image1_image2_matches.npz')
    keypoints_0 = matches_dict['keypoints0']
    keypoints_1 = matches_dict['keypoints1']
    
    #keypoints_0 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in keypoints_0]
    #keypoints_1 = [cv2.KeyPoint(x=float(pt[0]), y=float(pt[1]), size=1) for pt in keypoints_1]

    matches = matches_dict['matches']   
    
    pares_matches = np.empty([len(matches)], dtype=object)
    for i in range(len(matches)):
        if matches[i] == -1:
            pares_matches[i] = None
        else:
            pares_matches[i] = [keypoints_0[i], keypoints_1[matches[i]]]
    
      
    pares_matches = [np.array(pares_matches[i]) for i in range(len(pares_matches)) if pares_matches[i] is not None]

print('Super Glue')    
F_SuperGlue, best_matches_SG, best_inliers_SG = ransac_F(pares_matches, image_1, image_2, 3, 0.999, 0.2)


epipole_epipolarLines_click_plot(cv2.imread('image1.png'), F_SuperGlue, cv2.imread('image2.png'))


print('\n')


print('Sift') 

matches_sift, keypoints_sift_1, keypoints_sift_2 = launchNndrSift()

F_Sift, best_matches_Sift, best_inliers_Sift = ransac_F(matches_sift, image_1, image_2, 3, 0.999, 0.2)


epipole_epipolarLines_click_plot(cv2.imread('image1.png'), F_Sift, cv2.imread('image2.png'))
        


#imgMatched = cv2.drawMatches(cv2.imread('image_1.png'), keypoints_0, cv2.imread('image_2.png'), keypoints_1, best_matches,
#                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
            
#imgMatchedInliers = cv2.drawMatches(cv2.imread('image_1.png'), keypoints_0, cv2.imread('image_2.png'), keypoints_1, best_inliers,
#                                 None, flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS and cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
    
            
#plt.figure(figsize=(15, 10))
#plt.subplot(1, 2, 1)
#plt.title('Matches')
#plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)

# Mostrar los inliers
# plt.subplot(1, 2, 2)
# plt.title('Inliers with Hypothesis')
# plt.imshow(imgMatchedInliers, cmap='gray', vmin=0, vmax=255)
    
# plt.tight_layout()

#plt.imshow(imgMatched, cmap='gray', vmin=0, vmax=255)
# plt.draw()
# plt.waitforbuttonpress()


