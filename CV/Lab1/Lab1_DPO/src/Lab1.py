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
# Authors: David Padilla Orenga
#
#####################################################################################

import numpy as np
import matplotlib.pyplot as plt
import cv2

def assembly_transformation_matrix(R, t):
    """
        Creates a 4x4 transformation matrix from a rotation matrix (R) and a translation vector (t).
        - Inputs:
            · R (np.array): Rotation matrix (3x3).
            · t (np.array): Translation vector (3,).
        - Output:
            · np.array: Transformation matrix (4x4).
    """
    T = np.eye(4,4)
    T[0:3, 0:3] = R
    T[0:3, 3] = t
    return T

def assembly_camera_matrix(fx, fy, xo, yo):
    """
        Creates a camera intrinsic matrix (3x3).
        - Inputs:
            · fx (float): Focal length in x direction.
            · fy (float): Focal length in y direction.
            · xo (float): x-coordinate of the optical center.
            · yo (float): y-coordinate of the optical center.
        - Output:
            · np.array: Camera intrinsic matrix (3x3).
    """
    K = np.array([[fx, 0, xo],
                [0, fy, yo],
                [0, 0, 1]])
    return K

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

def project_to_camera(X_h, P):
    x_h_projected = P @ X_h
    x_h_projected /= x_h_projected[-1]
    return x_h_projected

def plot_point(ax, points, color='red', marker='+',labels=None):
    if points.ndim == 1:
        points = points[:, np.newaxis]
    x_coords, y_coords = points[0], points[1]

    for i in range(len(x_coords)):
        label = labels[i] if labels else None
        ax.scatter(x_coords[i], y_coords[i], color=color, marker=marker)
        if label:
            ax.text(x_coords[i], y_coords[i], '  ' + label, verticalalignment='center', color=color)

def get_limit_line(start_point, end_point, width, height):
    m = (end_point[1] - start_point[1]) / (end_point[0] - start_point[0])
    c = start_point[1] - m * start_point[0]
    y_left = m * 0 + c
    y_right = m * width + c
    x_top = (0 - c) / m
    x_bottom = (height - c) / m 
    points = np.array([
        [0, y_left],
        [width, y_right],
        [x_top, 0],
        [x_bottom, height]
    ])
    
    valid_points = points[(points[:, 0] >= 0) & (points[:, 0] <= width) &
                          (points[:, 1] >= 0) & (points[:, 1] <= height)]
    
    return valid_points[0], valid_points[1]

def get_limit_line_direction(direction, width, height):
    m = (direction[1]) / (direction[0])
    c = direction[1] - m * direction[0]
    y_left = m * 0 + c
    y_right = m * width + c
    x_top = (0 - c) / m
    x_bottom = (height - c) / m 
    points = np.array([
        [0, y_left],
        [width, y_right],
        [x_top, 0],
        [x_bottom, height]
    ])
    
    valid_points = points[(points[:, 0] >= 0) & (points[:, 0] <= width) &
                          (points[:, 1] >= 0) & (points[:, 1] <= height)]
    
    return valid_points[0], valid_points[1]
    
def plot_line(ax, start_point=None, end_point=None, direction=None, width=None, height=None, style='-', color='green'):
    if direction is not None:
        # Calculate points based on direction and length
        line_start, line_end = get_limit_line_direction(direction, width, height)
    elif end_point is not None:
        # Use the provided end point
        line_start = start_point
        line_end = end_point
        if width is not None and height is not None:
            line_start, line_end = get_limit_line(start_point, end_point, width, height)
    else:
        raise ValueError("Either an two points or a direction must be provided")

    
    ax.plot([line_start[0], line_end[0]], [line_start[1], line_end[1]], style, color=color)


if __name__ == '__main__':
    np.set_printoptions(precision=1,linewidth=1024,suppress=True)
    
    # Initialize camera and world parameters
    R_w_C1 = np.loadtxt('./Lab1_DPO/data/R_w_c1.txt')
    R_w_C2 = np.loadtxt('./Lab1_DPO/data/R_w_c2.txt')

    t_w_C1 = np.loadtxt('./Lab1_DPO/data/t_w_c1.txt')
    t_w_C2 = np.loadtxt('./Lab1_DPO/data/t_w_c2.txt')
    
    fx = 1165.723022
    fy = 1165.738037
    xo = 649.094971
    yo = 484.765015
    
    # Points in homogeneous coordinates
    xA_w_h = np.array([3.44, 0.80, 0.82, 1])
    xB_w_h = np.array([4.20, 0.80, 0.82, 1])
    xC_w_h = np.array([4.20, 0.60, 0.82, 1])
    xD_w_h = np.array([3.55, 0.60, 0.82, 1])
    xE_w_h = np.array([-0.01, 2.6, 1.21, 1]) 
    X_w_h = np.column_stack((xA_w_h, xB_w_h, xC_w_h, xD_w_h, xE_w_h))
    
    # Lines in homogeneous coordinates
    xAB_h = xB_w_h - xA_w_h
        
    # Projection matrixes
    T_C1_w = np.linalg.inv(assembly_transformation_matrix(R_w_C1, t_w_C1))
    T_C2_w = np.linalg.inv(assembly_transformation_matrix(R_w_C2, t_w_C2))
    K_C1 = assembly_camera_matrix(fx, fy, xo, yo)
    K_C2 = assembly_camera_matrix(fx, fy, xo, yo)
    P1 = get_projection_matrix(K_C1, T_C1_w)
    P2 = get_projection_matrix(K_C2, T_C2_w)

    # Project points from world to cameras coordinates
    x_C1_h = project_to_camera(X_w_h, P1)
    x_C2_h = project_to_camera(X_w_h, P2) 
    print("Homogenous coordinates of points for camera C1 (in px) are: \n",x_C1_h)
    print("Homogenous coordinates of points for camera C2 (in px) are: \n",x_C2_h)
    
    # Project lines from world to cameras coordinates
    x_AB_h = project_to_camera(xAB_h, P1)
    
    # Crear una sola figura con múltiples subplots
    img_C1 = cv2.cvtColor(cv2.imread("./Lab1_DPO/data/Image1.jpg"), cv2.COLOR_BGR2RGB)
    img_C2 = cv2.cvtColor(cv2.imread("./Lab1_DPO/data/Image2.jpg"), cv2.COLOR_BGR2RGB)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6))  # 1 fila, 2 columnas

    # Configuración del primer subplot
    ax1.set_xlabel('Coordinates X (píxeles)')
    ax1.set_ylabel('Coordinates Y (píxeles)')
    ax1.imshow(img_C1) 
    ax1.set_title('Image 1')
    x_C1_labels = ['a', 'b', 'c', 'd', 'e']
    plot_point(ax1, x_C1_h, labels=x_C1_labels) 
    plot_line(ax1, start_point=x_C1_h[:,2], end_point=x_C1_h[:,3], width=(xo*2), height=(yo*2))
    plot_line(ax1, start_point=x_C1_h[:,0], end_point=x_C1_h[:,1], width=(xo*2), height=(yo*2))
    plot_line(ax1, direction=x_AB_h, width=(xo*2), height=(yo*2))
    
    # Segundo subplot para la segunda imagen
    ax2.set_xlabel('Coordinates X (píxeles)')
    ax2.set_ylabel('Coordinates Y (píxeles)')
    ax2.imshow(img_C2)
    x_C2_labels = ['a', 'b', 'c', 'd', 'e']
    plot_point(ax2, x_C2_h, color='red', labels=x_C2_labels)
    plot_line(ax2, start_point=x_C2_h[:,2], end_point=x_C2_h[:,3], width=(xo*2), height=(yo*2))

    ax2.set_title('Image 2')
    plt.draw() 

    print('Click in the image to continue...')
    plt.waitforbuttonpress()
    