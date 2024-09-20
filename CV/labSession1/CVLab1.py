# from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
import numpy as np
import cv2

if __name__ == '__main__':
    
    # # Numpy function to set 
    np.set_printoptions(precision=4,linewidth=1024,suppress=True)


    #Load variables
    R_w_c1 = np.loadtxt('R_w_c1.txt')
    R_w_c2 = np.loadtxt('R_w_c2.txt')
    t_w_c1 = np.loadtxt('t_w_c1.txt')
    t_w_c2 = np.loadtxt('t_w_c2.txt')
    k = np.loadtxt('K.txt')

    # Esto es X_a_w. Hay que tener X_a_c
    X_a = np.array([3.44,0.80,0.82,1])
    X_a  = X_a.reshape(-1,1)

    X_b = np.array([4.20,0.80,0.82,1])
    X_b  = X_b.reshape(-1,1)


    # Compute the transpose of t_w_c1
    t_w_c1_transpose = t_w_c1.reshape(-1,1)
    
    # [R|t]
    R_t_w_c1 = np.hstack((R_w_c1,t_w_c1_transpose))

    # This is not necessary in the projection. Just only for 3D Transformations
    # R_t_w_c1 = np.vstack((R_t_w_c1,[0,0,0,1]))
    

    # P1 = K R_t_w_c1
    p1 = np.dot(k,R_t_w_c1)


    # P2 = K [R_w_c2|t_w_c2] (inline)
    p2 = np.dot(k,np.hstack((R_w_c2,t_w_c2.reshape(-1,1))))

    # X_a
    X_a2D = np.dot(p1,X_a)
    X_a2D = X_a2D/X_a2D[2]
    print(X_a2D)

    # X_b
    X_b2D = np.dot(p1,X_b)
    X_b2D = X_b2D/X_b2D[2]
    print(X_b2D)