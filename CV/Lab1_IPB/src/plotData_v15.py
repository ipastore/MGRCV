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
# Authors: Jesus Bermudez, Richard Elvira, Jose Lamarca, JMM Montiel
#
# Version: 1.5
#
#####################################################################################

from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np
import cv2

# Ensamble T matrix
def ensamble_T(R_w_c, t_w_c) -> np.array:
    """
    Ensamble the a SE(3) matrix with the rotation matrix and translation vector.
    """
    T_w_c = np.zeros((4, 4))
    T_w_c[0:3, 0:3] = R_w_c
    T_w_c[0:3, 3] = t_w_c
    T_w_c[3, 3] = 1
    return T_w_c

def plotLabeledImagePoints(x, labels, strColor,offset):
    """
        Plot indexes of points on a 2D image.
         -input:
             x: Points coordinates.
             strColor: Color of the text.
             offset: Offset from the point to the text.
         -output: None
         """
    for k in range(x.shape[1]):
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], labels[k], color=strColor)

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
        plt.text(x[0, k]+offset[0], x[1, k]+offset[1], str(k), color=strColor)

def plotLabelled3DPoints(ax, X, labels, strColor, offset):
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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], labels[k], color=strColor)

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
        ax.text(X[0, k]+offset[0], X[1, k]+offset[1], X[2,k]+offset[2], str(k), color=strColor)

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

def get_projection_from_Rtwc_toXc(X, R_w, t_w, K_c, point=True) -> np.array:
    """
    Get the projection of a 3D point in the world frame to the camera frame given the rotation matrix, translation vector (world to camera)and camera matrix.
    """
    
    #0 Get homogeneous coordinates for 3D points in world frame
    #Transpose
    X_h = X.reshape(1,-1).T

    if point:
        # Concatenate 1
        X_h = np.vstack((X_h, np.ones((1, 1))))
    else:
        # Concatenate 0
        X_h = np.vstack((X_h, np.zeros((1, 1))))    
    
    #1 Ensamble the R T matrix
    R_t_w_c = ensamble_T(R_w, t_w)
    #2 Inverse of R T matrix
    R_t_c_w = np.linalg.inv(R_t_w_c)
    #3 multiply by eye 3x4 matrix
    R_t_c_w = np.dot(np.eye(3,4), R_t_c_w)

    #4 Multiply by K matrix: This is exactli the Projection matrix
    K_R_t_c_w = np.dot(K_c, R_t_c_w)

    #5 T_c_w: Multiply by 3D point
    X_c = np.dot(K_R_t_c_w, X_h)

    #6 Normalize
    X_c = X_c / X_c[2]

    return X_c[:2]

def plot_line_from_projection_points(X, coor1, coor2, img, color):
    """
    Plot a line in an image given two points from a matrix containing the points.
    coor1 and coor2 are the indexes of the points in the matrix.
    img is the image to plot the line.
    color is the color of the line.
    """
    #Pass to homogeneous coordinates
    X = np.vstack((X, np.ones((1, 5))))
    l_ab = np.cross(X[:,coor1], X[:,coor2])
    #Normalize the line
    l_ab = l_ab / l_ab[2]

    #Plot the line
    plt.imshow(img)
    plt.plot([0, img.shape[1]], [-l_ab[2]/l_ab[1], -(l_ab[0]*img.shape[1]+l_ab[2])/l_ab[1]], color)
    # Set x and y limits to the image boundaries
    plt.xlim([0, img.shape[1]])
    plt.ylim([img.shape[0], 0])
    plt.show()

if __name__ == '__main__':

#region ############## Data ################
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)

    # Load ground truth
    R_w_c1 = np.loadtxt('./Lab1_IPB/data/R_w_c1.txt')
    R_w_c2 = np.loadtxt('./Lab1_IPB/data/R_w_c2.txt')

    t_w_c1 = np.loadtxt('./Lab1_IPB/data/t_w_c1.txt')
    t_w_c2 = np.loadtxt('./Lab1_IPB/data/t_w_c2.txt')

    K_c = np.loadtxt('./Lab1_IPB/data/K.txt')

    # Ensamble T matrix
    T_w_c1 = ensamble_T(R_w_c1, t_w_c1)
    T_w_c2 = ensamble_T(R_w_c2, t_w_c2)

    # 3D points in the world frame to transpose
    X_A = np.array([3.44, 0.80, 0.82])
    X_B = np.array([4.20, 0.80, 0.82])
    X_C = np.array([4.20, 0.60, 0.82])
    X_D = np.array([3.55, 0.60, 0.82]) 
    X_E = np.array([-0.01, 2.6, 1.21])
#endregion

#region # Excersize 1:
    # Make list or array of 3D points to insert in the function
    X_w_array = np.array([X_A, X_B, X_C, X_D, X_E])

    X_c1 = []
    X_c2 = []

    for i in range(len(X_w_array)):
        X_c1.append(get_projection_from_Rtwc_toXc(X_w_array[i], R_w_c1, t_w_c1, K_c))
        X_c2.append(get_projection_from_Rtwc_toXc(X_w_array[i], R_w_c2, t_w_c2, K_c))
#endregion

#region ################# DRAW 3D ###########################
    
    # # Example of transpose (need to have dimension 2)  and concatenation in numpy
    # X_w = np.vstack((np.hstack((np.reshape(X_A,(3,1)), np.reshape(X_B,(3,1)), np.reshape(X_C,(3,1)), np.reshape(X_D,(3,1)),np.reshape(X_E,(3,1)))), np.ones((1, 5))))

    # ##Plot the 3D cameras and the 3D points
    # fig3D = plt.figure(3)

    # ax = plt.axes(projection='3d', adjustable='box')
    # ax.set_xlabel('X')
    # ax.set_ylabel('Y')
    # ax.set_zlabel('Z')

    # drawRefSystem(ax, np.eye(4, 4), '-', 'W')
    # drawRefSystem(ax, T_w_c1, '-', 'C1')
    # drawRefSystem(ax, T_w_c2, '-', 'C2')

    # ax.scatter(X_w[0, :], X_w[1, :], X_w[2, :], marker='.')
    # # plotNumbered3DPoints(ax, X_w, 'r', (0.1, 0.1, 0.1)) # For plotting with numbers (choose one of the both options)
    # plotLabelled3DPoints(ax, X_w, ['A','B','C','D','E'], 'r', (-0.3, -0.3, 0.1)) # For plotting with labels (choose one of the both options)

    # #Matplotlib does not correctly manage the axis('equal')
    # xFakeBoundingBox = np.linspace(0, 4, 2)
    # yFakeBoundingBox = np.linspace(0, 4, 2)
    # zFakeBoundingBox = np.linspace(0, 4, 2)
    # plt.plot(xFakeBoundingBox, yFakeBoundingBox, zFakeBoundingBox, 'w.')

    # #Drawing a 3D segment A to B
    # draw3DLine(ax, X_A, X_B, '--', 'k', 1)

    # #Drawing a 3D segment C to D
    # draw3DLine(ax, X_C, X_D, '--', 'k', 1)

    # print('Close the figure to continue. Left button for orbit, right button for zoom.')
    # plt.show()
#endregion #################################################

#region ################# 2D plotting ##############
    #Load images 
    img1 = cv2.cvtColor(cv2.imread("./Lab1_IPB/data/Image1.jpg"), cv2.COLOR_BGR2RGB)
    img2 = cv2.cvtColor(cv2.imread("./Lab1_IPB/data/Image2.jpg"), cv2.COLOR_BGR2RGB)

    # Make numpy array to reshape and plot
    X_c1 = np.array(X_c1)
    X_c2 = np.array(X_c2)

    # Reshape X_c1 to remove the singleton dimension to Plot
    X_c1 = X_c1.reshape(5, 2)
    X_c2 = X_c2.reshape(5, 2) 

    # Transpose X_c1 to get the correct shape for plotting
    X_c1 = X_c1.T
    X_c2 = X_c2.T

    # Plot image 1
    plt.figure(1)
    plt.imshow(img1)
    plt.plot(X_c1[0,:],X_c1[1,:], '+r', markersize=15)
    plotLabeledImagePoints(X_c1, ['a','b','c','d','e'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    # plotNumberedImagePoints(X_c1, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 1')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()

    # Plot image 2 
    plt.figure(2)
    plt.imshow(img2)
    plt.plot(X_c1[0,:],X_c1[1,:], '+r', markersize=15)
    plotLabeledImagePoints(X_c1, ['a','b','c','d','e'], 'r', (20,-20)) # For plotting with labels (choose one of the both options)
    # plotNumberedImagePoints(X_c1, 'r', (20,25)) # For plotting with numbers (choose one of the both options)
    plt.title('Image 2')
    plt.draw()  # We update the figure display
    print('Click in the image to continue...')
    plt.waitforbuttonpress()
#endregion

#region ################# Excersize 2 ##############

# 2.5. Project  the  point  AB_inf  with  matrix  P    to  obtain  the  corresponding  vanishing 


    # #2.1: Line l_ab
    # plot_line_from_projection_points(X_c1, 0, 1, img1, 'g')
    # plot_line_from_projection_points(X_c2, 0, 1, img2, 'g')
    
    # #2.2: Line l_cd
    # plot_line_from_projection_points(X_c1, 2, 3, img1, 'g')
    # plot_line_from_projection_points(X_c2, 2, 3, img2, 'g')

    # 2.3. Compute p_12 the intersection point of l_ab and l_cd in image 1 and image 2.

    #Pass to homogeneous coordinates
    X_c1 = np.vstack((X_c1, np.ones((1, 5))))
    X_c2 = np.vstack((X_c2, np.ones((1, 5))))

    l_ab_1 = np.cross(X_c1[:,0], X_c1[:,1])
    l_cd_1 = np.cross(X_c1[:,2], X_c1[:,3])

    p_12_1 = np.cross(l_ab_1, l_cd_1)
    p_12_1 = p_12_1 / p_12_1[2]

    l_ab_2 = np.cross(X_c2[:,0], X_c2[:,1])
    l_cd_2 = np.cross(X_c2[:,2], X_c2[:,3])

    p_12_2 = np.cross(l_ab_2, l_cd_2)
    p_12_2 = p_12_2 / p_12_2[2]

    #Plot the intersection point
    plt.imshow(img1)
    plt.plot(p_12_1[0], p_12_1[1], 'or', markersize=5)
    plt.show()

    plt.imshow(img2)
    plt.plot(p_12_2[0], p_12_2[1], 'or', markersize=5)
    plt.show()

    # 2.4. Compute  the  3D  infinite  point  corresponding  to  the  3D  direction  defined  by points A and B, AB_inf. 

    AB_inf = X_B - X_A

    # 2.5. Project  the  point  AB_inf  with  matrix  P    to  obtain  the  corresponding  vanishing

    AB_inf_c1 = get_projection_from_Rtwc_toXc(AB_inf, R_w_c1, t_w_c1, K_c, point=False)
    AB_inf_c2 = get_projection_from_Rtwc_toXc(AB_inf, R_w_c2, t_w_c2, K_c, point=False)

    # Plot the vanishing points in image 1 and 2

    plt.imshow(img1)
    plt.plot(AB_inf_c1[0], AB_inf_c1[1], 'or', markersize=5)
    plt.show()

    plt.imshow(img2)
    plt.plot(AB_inf_c2[0], AB_inf_c2[1], 'or', markersize=5)
    plt.show()

    







#endregion

   

#endregion