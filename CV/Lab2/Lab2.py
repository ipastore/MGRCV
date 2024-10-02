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
#           Nacho pastor
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

def plot_epipolar_line(F, x1, ax2, img2):
    """
    Given a fundamental matrix and a point in image 1, plot the corresponding epipolar line in image 2.
    - F: Fundamental matrix (3x3)
    - x1: Point in image 1 (homogeneous coordinates, shape (3,))
    - ax2: Axis for image 2 where the epipolar line will be plotted
    - img2: Image 2 (for replotting as background)
    """
    # Compute the epipolar line in image 2
    l2 = F @ x1  # l2 = [a, b, c], where the line equation is ax + by + c = 0
    
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

def on_click(event, F, ax1, ax2, img2):
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
        plot_epipolar_line(F, x1_clicked, ax2, img2)

    

### MAIN ###

if __name__ == '__main__':
    
    np.set_printoptions(precision=1,linewidth=1024,suppress=True)
    
    try:
        #region
        # Initialize camera and world parameters
        T_w_c1 = load_matrix('./T_w_c1.txt')
        T_w_c2 = load_matrix('./T_w_c2.txt')
        T_C1_w = np.linalg.inv(T_w_c1)
        T_C2_w = np.linalg.inv(T_w_c2)
        K_C = load_matrix('./K_c.txt')  

        # Compute projection matrices
        P1 = get_projection_matrix(K_C, T_C1_w)
        P2 = get_projection_matrix(K_C, T_C2_w)
        
        # Loading set of points already projected
        x1_data = load_matrix('./x1Data.txt')  # 2D points from image 1
        x2_data = load_matrix('./x2Data.txt')  # 2D points from image 2
        x1 = np.array([[362, 104]])
        x2 = np.array([[304, 170]])
        X_h = triangulate_points(x1_data, x2_data, P1, P2)
        X_w_sol = load_matrix('./X_w.txt')

        print("Projection Matrix P1:")
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
        print('Close the figure to continue. Left button for orbit, right button for zoom.')
        plt.show()
        #endregion
        
        ## SECTION 2 ##
        F_21_test = load_matrix('./F_21_test.txt')
        
        ## 2D plotting example
        img1 = cv2.cvtColor(cv2.imread('image1.png'), cv2.COLOR_BGR2RGB)
        img2 = cv2.cvtColor(cv2.imread('image2.png'), cv2.COLOR_BGR2RGB)
        
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
        ax1.set_title('Image 2 - Epipolar Lines')
        
        # Connect the click event on image 1 to the handler
        fig.canvas.mpl_connect('button_press_event', lambda event: on_click(event, F_21_test , ax1, ax2, img2))
        
        print('Close the figure to continue. Select a point from Img1 to get the equivalent epipolar line.')
        plt.show()

        
    except Exception as e:
        print(f"An error occurred: {e}")