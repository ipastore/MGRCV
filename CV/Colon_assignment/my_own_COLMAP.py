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
    FIRST_TRIANGULATION_FLAG = False
    # TODO: Fix warning
    # WARNING: If disable, in-memory data structure won´t be initialized
    PNP_FLAG = True
    TRIANGULATE_3_CAMERAS_FLAG = True
    FULL_BUNDLE_3_CAMERAS = True
    VISUALIZATION_FULL_BUNDLE_3CAMERAS_FLAG = False
    ALL_VIZ_FLAG = False
    PNP_GT_FLAG = False

    # Define project structure
    project_root_dir = "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/own_projects"
    seq_name = "Seq_035"
    type = "toy"
    db_name = "database.db"
    project_dir = os.path.join(project_root_dir, seq_name, type)
    database_path = os.path.join(project_dir,db_name)
    cache_dir = os.path.join(project_dir, "cache")

    # Create cache directory if it does not exist
    if not os.path.exists(cache_dir): 
        os.makedirs(cache_dir)

    # List of image name (in order as COLMAP: 22,3,2,5 = 4,2,1,3 in toy database)
    # TODO: add a logic to select images with most matches
    image_names = ["f_0082926", "f_0082492", "f_0082452","f_0082558"]
    # image_names = [ "f_0082492", "f_0082558","f_0082452", "f_0082926"]
    # image_names =["f_0082558","f_0082452", "f_0082926", "f_0082492"]
    # image_names =["f_0082452", "f_0082926", "f_0082492", "f_0082558"]

    #IDs are assigned in alphabetical order
    # c1 = 1   # f_0082452
    # c2 = 2   # f_0082492
    # c3 = 3   # f_0082558
    # c4 = 4   # f_0082926

    # c_ids
    c_ids = [4, 2, 1, 3]
    
    # Same convention as IDs
    image_dir = os.path.join(project_dir, "images")
    image4 = cv2.imread(os.path.join(image_dir, f"{image_names[0]}.png"))
    image2 = cv2.imread(os.path.join(image_dir, f"{image_names[1]}.png"))
    image1 = cv2.imread(os.path.join(image_dir, f"{image_names[2]}.png"))
    image3 = cv2.imread(os.path.join(image_dir, f"{image_names[3]}.png"))

    images_list = [image4, image2, image1, image3]


    # Get camera K matrix. We only assume one camera for now with a pinhole model
    K = get_camera_intrinsics(database_path)
    # debug_print("Camera intrinsic matrix (K):")
    # debug_print(K)
    # debug_print(f"\n")


    #Toy example for seq 035 with images 2,3,5 and 22
    # Order of images to be added: 22 and 3 to initialize, add 2 and finall add 5
    # Database is already initialized
    # Cameras are already added

    # Step 0: Build a correspondence_graph
    images_info, adjacency = build_correspondence_graph(database_path)
    print(f"In-memory adjacency loaded: {len(images_info)} images.")

    
    match_list_1_2 = adjacency[c_ids[0]][c_ids[1]]   # kp1 --> c4, kp2 --> c2

    if FIRST_TRIANGULATION_FLAG: 

        # x1 --> c4, x2 --> c2
        x1 = []
        x2 = []
        for (kp1, kp2)in match_list_1_2:
            (r1, c1) = images_info[c_ids[0]]["keypoints"][kp1]
            (r2, c2) = images_info[c_ids[1]]["keypoints"][kp2]
            x1.append([c1, r1])  # (x, y)
            x2.append([c2, r2])

        ## x1 = x1, x2 = x2
        x1 = np.array(x1, dtype=np.float64)  # shape (2, N)
        x2 = np.array(x2, dtype=np.float64)  # shape (2, N)

        # Step 1 Extract R and t from F of images 4 and 23. Where 4 is the reference image as c1 and 2 is c2. world frame = c1 frame
        # image_names[0] = 4, image_names[1] = 2
        # Inside the function the F is transposed to get F_c2_c1
        R_c2_c1_option1, R_c2_c1_option2, t_c2_c1, F21= extract_R_t_from_F(database_path, image_names[0], image_names[1], K)

        # Retrieve 2D points from images 22 and 3 as x1 and x2
        #OLD with BUG: sqlite: now with adjancency in memory.
        # x1_old, x2_old = retrieve_matched_points_with_pair_id(database_path, image_names[0], image_names[1])
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

        ######################################## POSSIBLE BUG ########################################
        #BUG: check how to select valid pose, getting negative depths in camera 2. for the first example, when plotting demonstrates the contrary
        R_c2_c1_initial, t_c2_c1_initial, X_c1_initial = select_correct_pose_flexible(x1_h, x2_h, K, K, R_c2_c1_option1, R_c2_c1_option2, t_c2_c1)
        
        if R_c2_c1_initial is None:
            raise ValueError("No valid pose found")
        

        # Visualize residuals between 2D points and reprojection of 3D points
        #From World to image4 that is c1
        P_c1_c1_initial = get_projection_matrix(K, np.eye(4))
        x1_proj_initial = project_to_camera(P_c1_c1_initial, X_c1_initial)

        #From World to image2
        T_c2_c1_initial = ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
        P_c2_c1_initial = get_projection_matrix(K,T_c2_c1_initial)
        x2_proj_initial = project_to_camera(P_c2_c1_initial, X_c1_initial)

        visualize_epipolar_lines(F21, images_list[0], images_list[1], show_epipoles=False)

        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(images_list[0], x1, x1_proj_initial, "Initial Residuals in Image 4", ax=axs[0])
        visualize_residuals(images_list[1], x2, x2_proj_initial, 'Initial Residuals in Image 2', ax=axs[1])
        plt.tight_layout()
        plt.show()


        #Step 2 Full bundle adjustment
        # Initial guess for optimization
        theta_c2_c1_initial = crossMatrixInv(logm(R_c2_c1_initial.astype('float64')))
        t_norm = np.linalg.norm(t_c2_c1_initial, axis=-1)
        t_theta = np.arccos(t_c2_c1_initial[2]/t_norm)
        t_phi = np.arctan2(t_c2_c1_initial[1], t_c2_c1_initial[0])
        intial_guess = np.hstack((theta_c2_c1_initial, t_theta, t_phi, X_c1_initial[:3, :].flatten())) 
        nPoints = X_c1_initial.shape[1]

        # Least squares optimization
        optimized = least_squares(resBundleProjection, 
                            intial_guess, 
                            args=(x1, x2, K, nPoints),
                            method='lm'
                            )
            
        # Extract optimized parameters
        theta_c2_c1_opt = optimized.x[:3]
        # t_c2_c1_opt = optimized.x[3:5]
        t_theta = optimized.x[3]
        t_phi = optimized.x[4]
        t_c2_c1_opt = np.array([np.sin(t_theta)*np.cos(t_phi), np.sin(t_theta)*np.sin(t_phi), np.cos(t_theta)])
        X_c1_opt = optimized.x[5:].reshape(3, -1).T
        save_matrix(os.path.join(cache_dir, "theta_c2_c1_opt.txt"), theta_c2_c1_opt)
        save_matrix(os.path.join(cache_dir, "t_c2_c1_opt.txt"), t_c2_c1_opt)
        
        ## Visualize optimization
        R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
        T_c2_c1_opt = ensamble_T(R_c2_c1_opt, t_c2_c1_opt)
        P_c2_c1_opt = get_projection_matrix(K, T_c2_c1_opt)
        X_c1_opt_h = np.vstack((X_c1_opt.T, np.ones((1, X_c1_opt.shape[0]))))
        x1_proj_opt = project_to_camera(P_c1_c1_initial, X_c1_opt_h)
        x2_proj_opt = project_to_camera(P_c2_c1_opt, X_c1_opt_h)

        # View residuals from optimized calculations
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(images_list[0], x1, x1_proj_opt, "Optimized Residuals in Image 4", ax=axs[0])
        visualize_residuals(images_list[1], x2, x2_proj_opt, 'Optimized Residuals in Image 2', ax=axs[1])
        plt.tight_layout()
        plt.show()

        # Initial T 
        T_c1_c2_initial = np.linalg.inv(T_c2_c1_initial)
        
        #Opt T
        T_c1_c2_opt = np.linalg.inv(T_c2_c1_opt)
        
        # Create a 3D plot to compare all results
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        visualize_3D_comparison(
        ax,
        T_c1_c2_initial, X_c1_initial, # Initial estimate
        T_c1_c2_opt, X_c1_opt_h          # Optimized solution
        )

        visualize_3D_c1_2cameras(T_c1_c2_opt,X_c1_opt_h)

        # save the matrix in cache
        T_c1_c2_opt_path = os.path.join(cache_dir, "T_c1_c2_opt.txt")
        X_c1_opt_h_path = os.path.join(cache_dir, "X_opt_h.txt")
        save_matrix(T_c1_c2_opt_path, T_c1_c2_opt)
        save_matrix(X_c1_opt_h_path, X_c1_opt_h)
    
    if PNP_FLAG:

        # Loading from cache
        #### WARNING: little change of name. Now X_c4_opt_h is X_opt_h
        T_c1_c2_opt_path = os.path.join(cache_dir, "T_c1_c2_opt.txt")
        X_c1_opt_h_path = os.path.join(cache_dir, "X_opt_h.txt")
        T_c1_c2_opt = load_matrix(T_c1_c2_opt_path)
        X_opt_h = load_matrix(X_c1_opt_h_path)

        #Step3  Update database to keep track of 3D points
        insert_3d_points_in_memory_and_db(database_path, images_info,  X_opt_h, match_list_1_2, c_ids[0], c_ids[1])
        print("Triangulated points added to the database")

        #Step 4 Add image 2 and retrieve 3D points already plotted, seen by image 2
        match_list_3_1 = adjacency[c_ids[2]][c_ids[0]] 
        match_list_3_2 = adjacency[c_ids[2]][c_ids[1]]  

        # For each match, see if kp1 or kp2 has a 3D point
        # Save x1 points for pnp to select the correct pose for the initial guess from the F31
        pnp_points_2d = []
        pnp_points_3d = []
        x1_for_pnp = []
        x2_for_pnp = []
        for (kp3, kp1) in match_list_3_1:
            if kp1 in images_info[c_ids[0]]["kp3D"]: 
                p3d_id = images_info[c_ids[0]]["kp3D"][kp1]
                if p3d_id is not None:
                    # Query its 3D coordinates from DB or store it in memory
                    X, Y, Z = get_3d_point_coordinates(database_path, p3d_id)
                    (r3, c3) = images_info[c_ids[2]]["keypoints"][kp3] 
                    (r1, c1) = images_info[c_ids[0]]["keypoints"][kp1]
                    pnp_points_2d.append([c3, r3])
                    pnp_points_3d.append([X, Y, Z])
                    x1_for_pnp.append([c1, r1])


        # Repeat for c3-c2
        for (kp3, kp2) in match_list_3_2:
            if kp2 in images_info[c_ids[1]]["kp3D"]:
                p3d_id = images_info[c_ids[1]]["kp3D"][kp2]
                if p3d_id is not None:
                    X, Y, Z = get_3d_point_coordinates(database_path, p3d_id)
                    (r3, c3) = images_info[c_ids[2]]["keypoints"][kp3]
                    (r2, c2) = images_info[c_ids[1]]["keypoints"][kp2]
                    pnp_points_2d.append([c3, r3])
                    pnp_points_3d.append([X, Y, Z])
                    x2_for_pnp.append([c2, r2])

        pnp_points_2d = np.array(pnp_points_2d, dtype=np.float64).T    # shape (2, N)
        x1_for_pnp = np.array(x1_for_pnp, dtype=np.float64).T        # shape (2, N)
        x2_for_pnp = np.array(x2_for_pnp, dtype=np.float64).T        # shape (2, N)
        pnp_points_3d = np.array(pnp_points_3d, dtype=np.float64)    # shape (N, 3)
        x3_for_pnp = pnp_points_2d
        save_matrix(os.path.join(cache_dir, "x1.txt"), x1_for_pnp)
        save_matrix(os.path.join(cache_dir, "x2.txt"), x2_for_pnp)
        save_matrix(os.path.join(cache_dir, "x3.txt"), x3_for_pnp)


        debug_print(f"2D points from {image_names[2]} ")
        debug_print(pnp_points_2d.shape)
        debug_print(f"3D points from {image_names[2]} ")
        debug_print(pnp_points_3d.shape)
        debug_print(f"\n")
        
         # 103 out of 109 points. 33 are for c1 and 76 for c2
        print(f"New camera sees {pnp_points_2d.shape[1]} 3D points out of {X_opt_h.shape[1]}")


        #Step 5 pnp for the 3rd camera
        # Own pnp is implemented as a linear optimization of the camera pose given seen 3D points and their 2D projections
        # Initial guess is important for the optimization
        R_c3_c1_option1, R_c3_c1_option2, t_c3_c1, F31= extract_R_t_from_F(database_path, image_names[0], image_names[2], K)

        # Convert x1_for_pnp  pnp_points_2d to homogeneous coordinates
        x1_for_pnp_h = np.vstack((x1_for_pnp, np.ones(x1_for_pnp.shape[1])))
        pnp_points_2d_h = np.vstack((pnp_points_2d, np.ones(pnp_points_2d.shape[1])))
        
        # Select correct initial pose
        # Computing the correct pose with x1_for_pnp_h.shape[1] = 33 points
        R_c3_c1_initial, t_c3_c1_initial, _ = select_correct_pose_flexible(x1_for_pnp_h, pnp_points_2d_h, K, K, R_c3_c1_option1, R_c3_c1_option2, t_c3_c1)
            
        if R_c3_c1_initial is None:
            raise ValueError("No valid pose found")
        
        #Convert R matrix to Rvec for initial guess of PnP
        rvec_c3_c1_initial = crossMatrixInv(logm(R_c3_c1_initial.astype('float64')))

        # hstack R and t to initial guess
        pnp_intial_guess = np.hstack((rvec_c3_c1_initial, t_c3_c1_initial))

        # # Solve own PnP
        rvec_c3_c1_pnp, t_c3_c1_pnp = own_PnP(pnp_points_3d, pnp_points_2d, K, pnp_intial_guess)
        print("Own PnP success:", rvec_c3_c1_pnp, t_c3_c1_pnp)
        print(f"\n")
        save_matrix(os.path.join(cache_dir, "rvec_c3_c1_pnp.txt"), rvec_c3_c1_pnp)
        save_matrix(os.path.join(cache_dir, "t_c3_c1_pnp.txt"), t_c3_c1_pnp)


        # Recover R matrix from rvec
        R_c3_c1_pnp = expm(crossMatrix(rvec_c3_c1_pnp))
        save_matrix(os.path.join(cache_dir, "R_c3_c1_pnp.txt"), R_c3_c1_pnp)

        # Ensemble T matrix
        T_c3_c1_pnp = ensamble_T(R_c3_c1_pnp, t_c3_c1_pnp)

        # Invert T matrix to get T_c1_c3 and plot
        T_c1_c3_pnp = np.linalg.inv(T_c3_c1_pnp)

        # Visualize 3D
        if ALL_VIZ_FLAG:
            visualize_3D_3cameras(T_c1_c2_opt,T_c1_c3_pnp,X_opt_h)

            ### POSSIBLE BUG:WARNING: WHen compairing with pnp of CV2 translation vector is huge
            if PNP_GT_FLAG: # Transpose 2D points for cv2
                points2d_for_pnp_cv2 = pnp_points_2d.T
                # PnP cv2 for comparison
                retval, rvec, tvec = cv2.solvePnP(
                    pnp_points_3d, 
                    points2d_for_pnp_cv2, 
                    K,  # Correct argument name
                    None,  # Distortion coefficients, 
                    flags=cv2.SOLVEPNP_EPNP 
                )
                if not retval:
                    print("PnP failed for camera 3!")
                else:
                    print("PnP success:", rvec, tvec)

                    # Recover R matrix from rvec
                R_pnp = expm(crossMatrix(rvec))
                # Ensemble T matrix
                T_pnp = ensamble_T(R_pnp, tvec)
                # Invert T matrix to get T_c1_c3 and plot
                T_pnp = np.linalg.inv(T_pnp)
                # Visualize 3D
                visualize_3D_3cameras(T_c1_c2_opt,T_pnp, X_opt_h, adjust_plot_limits=False)


    

    #Step 6 triangulate new points with pairs of image 2 and image 3; and image 2 and image 22 (the already added images)
    # Update database to keep track of 3D points

    if TRIANGULATE_3_CAMERAS_FLAG:
        # Load from cache
        R_c3_c1_pnp = load_matrix(os.path.join(cache_dir, "R_c3_c1_pnp.txt"))
        t_c3_c1_pnp = load_matrix(os.path.join(cache_dir, "t_c3_c1_pnp.txt"))
        x1_for_pnp = load_matrix(os.path.join(cache_dir, "x1.txt"))
        x2_for_pnp = load_matrix(os.path.join(cache_dir, "x2.txt"))
        x3_for_pnp = load_matrix(os.path.join(cache_dir, "x3.txt"))

        # Triangulate new points for pair c3-c1
        new_X_c1 = triangulate_new_points_for_pair(
            database_path, adjacency, images_info,
            c1_id = c_ids[0], c2_id=c_ids[2],
            R_c2_c1 = R_c3_c1_pnp,  
            t_c2_c1 = t_c3_c1_pnp,
            K=K, plot_residuals=True, img1 = images_list[0], img2 = images_list[2])
            

        # Visualize 3D
        all_triangulated_points_3d = get_all_3d_points(database_path)
        all_triangulated_points_3d = all_triangulated_points_3d.T
        if ALL_VIZ_FLAG:
            visualize_3D_3cameras(T_c1_c3_pnp,T_c1_c2_opt,all_triangulated_points_3d)

        
        # # Triangulate new points for pair c3-c2

        # # Get R_c3_c2
        R_c3_c2_option1, R_c3_c2_option2, t_c3_c2, F32= extract_R_t_from_F(database_path, image_names[1], image_names[2], K)

        # Convert x1_for_pnp  pnp_points_2d to homogeneous coordinates
        x2_for_pnp_h = np.vstack((x2_for_pnp, np.ones(x2_for_pnp.shape[1])))
        # Get the 3D points seen by camera 2, indexing after camera 1
        # TODO: not so robust logic, should be improved
        x3_for_triangulate_c2 = x3_for_pnp[:, :x2_for_pnp.shape[1]]
        x3_for_triangulate_c2 = np.vstack((x3_for_triangulate_c2, np.ones(x3_for_triangulate_c2.shape[1])))
        
        # Select correct initial pose
        # Computing the correct pose with x2_for_pnp_h.shape[1] =76 points
        R_c3_c2_initial, t_c3_c2_initial, _ = select_correct_pose_flexible(x2_for_pnp_h, x3_for_triangulate_c2, K, K, R_c3_c2_option1, R_c3_c2_option2, t_c3_c2, plot_FLAG=False)
            
        if R_c3_c2_initial is None:
            raise ValueError("No valid pose found")
        
        # Get new matches for c3-c2
        # WARNING: Points should be plotted in c1 frame
        # Should multiply new points by T_c1_c2_opt
        # TODO: FIX NAMING! of images they got mixed up often
        new_X_c1 = triangulate_new_points_for_pair(
            database_path, adjacency, images_info,
            c1_id = c_ids[1], c2_id=c_ids[2],
            R_c2_c1 = R_c3_c2_initial,  
            t_c2_c1 = t_c3_c2,
            K=K, plot_residuals=True, 
            img1 = images_list[1], img2 = images_list[2], T_c1_c2 = T_c1_c2_opt)
        
        all_triangulated_points_3d = get_all_3d_points(database_path)
        all_triangulated_points_3d = all_triangulated_points_3d.T
        if ALL_VIZ_FLAG:
            visualize_3D_3cameras(T_c1_c3_pnp,T_c1_c2_opt,all_triangulated_points_3d)


     
    # #Step 7 Full bundle adjustment with triangulated points and all cameras positions
    # # Plot optimized camera positions and 3D points

    if FULL_BUNDLE_3_CAMERAS:

        # Load from cache
        rvec_c3_c1_pnp = load_matrix(os.path.join(cache_dir, "rvec_c3_c1_pnp.txt"))
        t_c3_c1_pnp = load_matrix(os.path.join(cache_dir, "t_c3_c1_pnp.txt"))
        rvec_c2_c1_opt = load_matrix(os.path.join(cache_dir, "theta_c2_c1_opt.txt"))
        t_c2_c1_opt = load_matrix(os.path.join(cache_dir, "t_c2_c1_opt.txt"))
        
        # Buld a new data-structure for observation list list and 3d points dictionary
        # obs_list contains: img_id, pid, x_meas, y_meas
        # points_3d_dict contains: pid, X, Y, Z
        obs_list, points_3d_dict = load_observations_and_points(database_path)

        # Build a new data-structure to store the cameras poses. Assume camera 1 is fixed in the origin. All cameras share K intrinsics, loaded at start
        # TODO: to fully automate the process later for all cameras: I should store initial rvec and tvec in db in the two_view_geometries
        camera_data = {
            c_ids[0]: {
                "fixed": True,
                "rvec": np.zeros(3),  
                "tvec": np.zeros(3),
                "index": None         # not used if fixed
            },
            c_ids[1]: {
                "fixed": False,
                "rvec": rvec_c2_c1_opt,  
                "tvec": t_c2_c1_opt,
                "index": 0              # offset for indexing
            },
            c_ids[2]: {
                "fixed": False,
                "rvec": rvec_c2_c1_opt,
                "tvec": t_c2_c1_opt,
                "index": 1
            },
        }

         # Run BA
        result = run_incremental_ba(database_path, camera_data, obs_list, points_3d_dict, K)
        
    
    # if VISUALIZATION_FULL_BUNDLE_3CAMERAS_FLAG:
    #     #load from cache
    #     theta_c2_c1_opt = load_matrix('../data/cache/theta_c2_c1_opt.txt')
    #     t_c2_c1_opt = load_matrix('../data/cache/t_c2_c1_opt.txt')
    #     theta_c3_c1_opt = load_matrix('../data/cache/theta_c3_c1_opt.txt')
    #     t_c3_c1_opt = load_matrix('../data/cache/t_c3_c1_opt.txt')
    #     X_c1_opt = load_matrix('../data/cache/X_c1_w_opt.txt')

    #     # Recover results

    #     X_c1_w_opt_h = np.vstack((X_c1_opt, np.ones((1, X_c1_opt.shape[1]))))

    #     # Convert rotation vectors back to matrices for visualization
    #     R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
    #     R_c3_c1_opt = expm(crossMatrix(theta_c3_c1_opt))

    #     # Print results
    #     print("Optimized Camera 2 Rotation Vector:", theta_c2_c1_opt)
    #     print("Optimized Camera 2 Translation Vector:", t_c2_c1_opt)
    #     print("Optimized Camera 3 Rotation Angle (around y-axis):", theta_c3_c1_opt)
    #     print("Optimized Camera 3 Translation (along z-axis):", t_c3_c1_opt)

    #     # Initiliaze the visualization
    #     R_c2_c1_initial = expm(crossMatrix(theta_c2_c1_initial))
    #     T_c2_c1_initial = ensamble_T(R_c2_c1_initial,t_c2_c1_initial)
       
    #     T_c3_c1_initial = ensamble_T(R_c3_c1_initial,t_c3_c1_initial)
       
    #     T_c1_c2_initial = np.linalg.inv(T_c2_c1_initial)
    #     T_c1_c3_initial = np.linalg.inv(T_c3_c1_initial)

    #     R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
    #     T_c2_c1_opt = ensamble_T(R_c2_c1_opt,t_c2_c1_opt)

    #     R_c3_c1_opt = expm(crossMatrix(theta_c3_c1_opt))
    #     T_c3_c1_opt = ensamble_T(R_c3_c1_opt,t_c3_c1_opt)

    #     T_c1_c2_opt = np.linalg.inv(T_c2_c1_opt)
    #     T_c1_c3_opt = np.linalg.inv(T_c3_c1_opt)

    #     P_c1_c1 = get_projection_matrix(K, np.eye(4))
    #     P_c2_c1_opt = get_projection_matrix(K, T_c2_c1_opt)
    #     P_c3_c1_opt = get_projection_matrix(K, T_c3_c1_opt)

    #     x1_proj_opt = project_to_camera(P_c1_c1, X_c1_w_opt_h)
    #     x2_proj_opt = project_to_camera(P_c2_c1_opt, X_c1_w_opt_h)
    #     x3_proj_opt = project_to_camera(P_c3_c1_opt, X_c1_w_opt_h)

    #     # View residuals from optimized calculations
    #     fig, axs = plt.subplots(1, 3, figsize=(18, 6))
    #     visualize_residuals(image1, x1, x1_proj_opt, "Optimized Residuals in Image 1",ax=axs[0])
    #     visualize_residuals(image2, x2, x2_proj_opt, 'Optimized Residuals in Image 2',ax=axs[1])
    #     visualize_residuals(image3, x3, x3_proj_opt, 'Optimized Residuals in Image 3',ax=axs[2])
    #     plt.tight_layout()
    #     plt.show()





    #Step 8 Repeat steps 4 to 8 for the rest of the cameras. Some effort should be done to automate the process