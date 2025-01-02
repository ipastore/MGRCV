import os
import numpy as np
import sqlite3
import cv2

from utils.cv_plot_bundle_sqlite_helpers_functions import *

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

if __name__ == "__main__":

    #DEBUG FLAG
    DEBUG = True

    # Define project structure
    project_root_dir = "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/own_projects"
    seq_name = "Seq_035"
    type = "toy"
    db_name = "database.db"
    project_dir = os.path.join(project_root_dir, seq_name, type)
    database_path = os.path.join(project_dir,db_name)

    # List of image name (in order as COLMAP: 22,3,2,5)
    # TODO: add a logic to select images with most matches
    image_names = ["f_0082926", "f_0082492", "f_0082558","f_0082452"]
    
    image_dir = os.path.join(project_dir, "images")
    image1 = cv2.imread(os.path.join(image_dir, f"{image_names[0]}.png"))
    image2 = cv2.imread(os.path.join(image_dir, f"{image_names[1]}.png"))
    image3 = cv2.imread(os.path.join(image_dir, f"{image_names[2]}.png"))
    image4 = cv2.imread(os.path.join(image_dir, f"{image_names[3]}.png"))


    # Get camera K matrix. We only assume one camera for now with a pinhole model
    K = get_camera_intrinsics(database_path)
    # debug_print("Camera intrinsic matrix (K):")
    # debug_print(K)
    # debug_print(f"\n")


    #Toy example for seq 035 with images 2,3,5 and 22
    # Order of images to be added: 22 and 3 to initialize, add 2 and finall add 5
    # Database is already initialized
    # Cameras are already added

    # Step 1 Extract R and t from F of images 22 and 3. Where 22 is the reference image as c1 and 3 is c2. world frame = c1 frame
    # Inside the function the F is transposed to get F_c2_c1
    R_c2_c1_option1, R_c2_c1_option2, t_c2_c1, F21= extract_R_t_from_F(database_path, image_names[0], image_names[1], K)

    # Retrieve 2D points from images 22 and 3 as x1 and x2
    x1, x2 = retrieve_matched_points_with_pair_id(database_path, image_names[0], image_names[1])
    debug_print(f"2D points from {image_names[0]} ")
    debug_print(x1.shape)
    debug_print(f"2D points from {image_names[1]} ")
    debug_print(x2.shape)
    debug_print(f"\n")
    
    # Convert to column vectors and homogeneous coordinates for the camera points x1 to x1_h and x2 to x2_h
    x1 = x1.T
    x2 = x2.T
    x1_h = np.vstack((x1, np.ones(x1.shape[1])))
    x2_h = np.vstack((x2, np.ones(x2.shape[1])))
    debug_print(f"2D points of {image_names[0]} in homogeneous coordinates")
    debug_print(x1_h.shape)
    debug_print(f"2D points of {image_names[1]} in homogeneous coordinates")
    debug_print(x2_h.shape)
    debug_print(f"\n")
    
    # Select correct pose


    R_c2_c1_initial, t_c2_c1_initial, X_c1_initial = select_correct_pose_flexible(x1_h, x2_h, K, K, R_c2_c1_option1, R_c2_c1_option2, t_c2_c1)
    
    if R_c2_c1_initial is None:
        raise ValueError("No valid pose found")
    

    # Visualize residuals between 2D points and reprojection of 3D points
    #From World to image1
    P_c1_c1_initial = get_projection_matrix(K, np.eye(4))
    x1_proj_initial = project_to_camera(P_c1_c1_initial, X_c1_initial)

    #From World to image2
    T_c2_c1_initial = ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
    P_c2_c1_initial = get_projection_matrix(K,T_c2_c1_initial)
    x2_proj_initial = project_to_camera(P_c2_c1_initial, X_c1_initial)

    visualize_epipolar_lines(F21, image1, image2, show_epipoles=True)

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    visualize_residuals(image1, x1, x1_proj_initial, "Initial Residuals in Image 1", ax=axs[0])
    visualize_residuals(image2, x2, x2_proj_initial, 'Initial Residuals in Image 2', ax=axs[1])
    plt.tight_layout()
    plt.show()




    #Step 2 Triangulate points of image 22 and 
    # Update database to keep track of 3D points
    # Plot

    #Step 3 Full bundle adjustment
    # Plot


    #Step 4 Add image 2 and retrieve 3D points already plotted, seen by image 2

    #Step 5 pnp for image 2
    # Plot initial guess for camera position


    #Step 6 local bundle adjustment for cameras poistions
    # Plot optimized camera positions

    #Step 7 triangulate new points with pairs of image 2 and image 3; and image 2 and image 22 (the already added images)
    # Update database to keep track of 3D points

    #Step 8 Full bundle adjustment with triangulated points and all cameras positions
    # Plot optimized camera positions and 3D points

    #Step 9 Repeat steps 4 to 8 for image 5