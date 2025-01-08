import os
import numpy as np
import sqlite3
import cv2
import scipy


from utils.cv_plot_bundle_sqlite_helpers_functions import *

def debug_print(*args, **kwargs):
    if DEBUG:
        print(*args, **kwargs)

if __name__ == "__main__":

    #TODO:
    # Add camera structure reading from two_view_geometries
    # Update cameras relative poses to c1 in the database whenever calculating new pose
    # Filter points with high reprojection error


    #DEBUG FLAG
    DEBUG = True
    FIRST_TRIANGULATION_FLAG = True
    PNP_FLAG = True
    PNP_GT_FLAG = True
    TRIANGULATE_3_CAMERAS_FLAG = True
    FULL_BUNDLE_3_CAMERAS = True
    VISUALIZATION_FULL_BUNDLE_3CAMERAS_FLAG = True
    PNP_CAMERA4_FLAG = True
    TRIANGULATE_CAMERA_4_FLAG = True
    FULL_BUNDLE_4CAMERAS_FLAG = True
    FULL_BUNDLE_4CAMERAS_VISUALIZATION_FLAG = True
    ALL_VIZ_FLAG = True


    # Define project structure
    project_root_dir = "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/own_projects"
    # seq_name = "Seq_035"
    seq_name = "Seq_027"
    type = "more_flexible"
    db_name = "database.db"
    project_dir = os.path.join(project_root_dir, seq_name, type)
    database_path = os.path.join(project_dir,db_name)
    cache_dir = os.path.join(project_dir, "cache")

    # Create cache directory if it does not exist
    if not os.path.exists(cache_dir): 
        os.makedirs(cache_dir)

    # List of image name (in order as COLMAP: 22,3,2,5 = 4,2,1,3 in toy database)
    # TODO: add a logic to select images with most matches
    
    
    ############################ SEQ_035 ########################################
    # image_names = ["f_0082926", "f_0082492", "f_0082452","f_0082558"]
    # image_names = [ "f_0082492", "f_0082558","f_0082452", "f_0082926"]
    # image_names =["f_0082558","f_0082452", "f_0082926", "f_0082492"]
    # image_names =["f_0082452", "f_0082926", "f_0082492", "f_0082558"]

    #SEQ_035 IDs are assigned in alphabetical order
    # c1 = 1   # f_0082452
    # c2 = 2   # f_0082492
    # c3 = 3   # f_0082558
    # c4 = 4   # f_0082926

    # c_ids
    # c_ids = [4, 2, 1, 3]

    #Images are displayed in BGR. 
    # # Same convention as IDs
    # image_dir = os.path.join(project_dir, "images")
    # image4 = cv2.imread(os.path.join(image_dir, f"{image_names[0]}.png"))
    # image2 = cv2.imread(os.path.join(image_dir, f"{image_names[1]}.png"))
    # image1 = cv2.imread(os.path.join(image_dir, f"{image_names[2]}.png"))
    # image3 = cv2.imread(os.path.join(image_dir, f"{image_names[3]}.png"))

    # images_list = [image4, image2, image1, image3]
    ############################ SEQ_035 ########################################

    ############################ SEQ_027 ########################################
    image_names = ["f_0064636", "f_0064722", "f_0064886","f_0064962", "f_0064556"]


    #SEQ_035 IDs are assigned in alphabetical order
    # c1 = 1   # f_0064556
    # c2 = 2   # f_0064636
    # c3 = 3   # f_0064722
    # c4 = 4   # f_0064886
    # c5 = 5   # f_0064962

    # c_ids
    c_ids = [2, 3, 4, 5, 1]

    #Images are displayed in BGR. 
    # Same convention as IDs
    image_dir = os.path.join(project_dir, "images")
    image2 = cv2.imread(os.path.join(image_dir, f"{image_names[0]}.png"))
    image3 = cv2.imread(os.path.join(image_dir, f"{image_names[1]}.png"))
    image4 = cv2.imread(os.path.join(image_dir, f"{image_names[2]}.png"))
    image5 = cv2.imread(os.path.join(image_dir, f"{image_names[3]}.png"))
    image1 = cv2.imread(os.path.join(image_dir, f"{image_names[4]}.png"))

    images_list = [image2, image3, image4, image5, image1]

    ############################ SEQ_027 ########################################


    # Get camera K matrix. We only assume one camera for now with a pinhole model
    K = get_camera_intrinsics(database_path)
    debug_print("Camera intrinsic matrix (K):")
    debug_print(K)
    debug_print(f"\n")


    # Step 0: Build a correspondence_graph
    images_info, adjacency = build_correspondence_graph(database_path)
    print(f"In-memory adjacency loaded: {len(images_info)} images.")
    # TODO: automate for all cameras
    match_list_1_2 = adjacency[c_ids[0]][c_ids[1]]   
    match_list_3_1 = adjacency[c_ids[2]][c_ids[0]] 
    match_list_3_2 = adjacency[c_ids[2]][c_ids[1]]
    match_list_4_1 = adjacency[c_ids[3]][c_ids[0]]
    match_list_4_2 = adjacency[c_ids[3]][c_ids[1]]
    match_list_4_3 = adjacency[c_ids[3]][c_ids[2]]
    


    if FIRST_TRIANGULATION_FLAG: 
        
        # Get 2D points from the matches
        x1 = []
        x2 = []
        for (kp1, kp2)in match_list_1_2:
            (c1, r1) = images_info[c_ids[0]]["keypoints"][kp1]
            (c2, r2) = images_info[c_ids[1]]["keypoints"][kp2]
            x1.append([c1, r1])  # (x, y)
            x2.append([c2, r2])

        x1 = np.array(x1, dtype=np.float64)  # shape (2, N)
        x2 = np.array(x2, dtype=np.float64)  # shape (2, N)

        # Step 1 Extract R and t from 
        # Inside the function the F is transposed to get F_c2_c1. If the pair in the database is already transposed, 
        # the F is not transposed again to always return F21 of image_names[0](c1) to image_names[1](c2)
        R_c2_c1_option1, R_c2_c1_option2, t_c2_c1, F21= extract_R_t_from_F(database_path, image_names[0], image_names[1], K)

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
        # Filter the negative depths for X_c1_initial, x1_h, x2_h
        R_c2_c1_initial, t_c2_c1_initial, X_c1_initial, x1_h, x2_h, match_list_1_2 = select_correct_pose_flexible_and_filter(x1_h, x2_h, K, K, R_c2_c1_option1, 
                                                                                                             R_c2_c1_option2, t_c2_c1, match_list_1_2, plot_FLAG=True,
                                                                                                             filtering = True)
        
        if R_c2_c1_initial is None:
            raise ValueError("No valid pose found")
        
        # Visualize epipolar lines: x2 points in image 1
        # visualize_epipolar_lines(F21, images_list[1], images_list[0], show_epipoles=True)
        
        # Compute and visualize residuals
        # Filter is apply by calculating the percentile of residuals and keeping the best points for X_c1_initial, x1_h, x2_h
        X_c1_initial,x1_h, x2_h, match_list_1_2 = compute_residulas_and_filter(x1_h, x2_h, X_c1_initial, R_c2_c1_initial, t_c2_c1_initial,
                                                                                match_list=match_list_1_2, K=K,img1= images_list[0],
                                                                                img2= images_list[1], c_id_1=c_ids[0],c_id_2=c_ids[1],
                                                                                percentile_filter = 100, plot_FLAG = ALL_VIZ_FLAG)
                
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
                            args=(x1_h[:2,:], x2_h[:2,:], K, nPoints),
                            method= "lm",
                            verbose=2,
                            ftol=1e-2,
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

        # Filter by norm: tuning for Seq_027 (no filter)
        norms = np.linalg.norm(X_c1_opt[:, :3], axis=1)
        percentile_norm = np.percentile(norms, 100)
        filtered_indices = norms <= percentile_norm

        X_c1_opt = X_c1_opt[filtered_indices, :]
        match_list_1_2 = [match_list_1_2[i] for i in range(len(match_list_1_2)) if filtered_indices[i]]
        x1_h = x1_h[:, filtered_indices]
        x2_h = x2_h[:, filtered_indices]

        print(f"Filtered out {np.sum(~filtered_indices)} points with high norm.")

        
        ## Visualize optimization
        R_c2_c1_opt = expm(crossMatrix(theta_c2_c1_opt))
        T_c2_c1_opt = ensamble_T(R_c2_c1_opt, t_c2_c1_opt)
        P_c2_c1_opt = get_projection_matrix(K, T_c2_c1_opt)
        X_c1_opt_h = np.vstack((X_c1_opt.T, np.ones((1, X_c1_opt.shape[0]))))
        P_c1_c1_initial = get_projection_matrix(K, np.eye(4))
        x1_proj_opt = project_to_camera(P_c1_c1_initial, X_c1_opt_h)
        x2_proj_opt = project_to_camera(P_c2_c1_opt, X_c1_opt_h)

        # View residuals from optimized calculations
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        visualize_residuals(images_list[0], x1_h[:2,:], x1_proj_opt, "Optimized Residuals in Image 4", ax=axs[0], adjust_limits= False)
        visualize_residuals(images_list[1], x2_h[:2,:], x2_proj_opt, 'Optimized Residuals in Image 2', ax=axs[1], adjust_limits= False)
        plt.tight_layout()
        plt.show()

        # Initial T 
        T_c2_c1_initial = ensamble_T(R_c2_c1_initial, t_c2_c1_initial)
        T_c1_c2_initial = np.linalg.inv(T_c2_c1_initial)
        
        #Opt T
        T_c1_c2_opt = np.linalg.inv(T_c2_c1_opt)
        
        if ALL_VIZ_FLAG: # Create a 3D plot to compare all results
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
    
        # Filter and update database 
        X_opt_h = load_matrix(os.path.join(cache_dir, "X_opt_h.txt"))

        insert_3d_points_in_memory_and_db(database_path, images_info,  X_opt_h, match_list_1_2, c_ids[0], c_ids[1])
        print(f"Inserted {X_opt_h.shape[1]} 3D points in the database")
    
    if PNP_FLAG:

        # Loading from cache
        T_c1_c2_opt_path = os.path.join(cache_dir, "T_c1_c2_opt.txt")
        X_c1_opt_h_path = os.path.join(cache_dir, "X_opt_h.txt")
        T_c1_c2_opt = load_matrix(T_c1_c2_opt_path)
        X_opt_h = load_matrix(X_c1_opt_h_path)

        # For each match, see if kp1 or kp2 has a 3D point
        # Save x1 points for pnp to select the correct pose for the initial guess from the F31
        pnp_points_2d = []
        pnp_points_3d = []
        x1_for_pnp = []
        x2_for_pnp = []


        pnp_points_2d, pnp_points_3d, x1_for_pnp = get_points_seen_by_camera(database_path, images_info, c_ids[2],c_ids[0], 
                                                        match_list_3_1, pnp_points_2d, pnp_points_3d, x1_for_pnp)

        pnp_points_2d, pnp_points_3d, x2_for_pnp = get_points_seen_by_camera(database_path, images_info, c_ids[2],c_ids[1],match_list_3_2,
                                                        pnp_points_2d, pnp_points_3d, x2_for_pnp)

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
        
        
        print(f"New camera sees {pnp_points_2d.shape[1]} 3D points out of {X_opt_h.shape[1]}")


        #Step 5 pnp for the 3rd camera
        # Own pnp is implemented as a least square optimization of the camera pose given seen 3D points and their 2D projections
        # Initial guess is important for the optimization. The pose is calculated from camera 1 frame
        R_c3_c1_option1, R_c3_c1_option2, t_c3_c1, F31= extract_R_t_from_F(database_path, image_names[0], image_names[2], K)

        # Convert x1_for_pnp  pnp_points_2d to homogeneous coordinates
        x1_for_pnp_h = np.vstack((x1_for_pnp, np.ones(x1_for_pnp.shape[1])))
        pnp_points_2d_h = np.vstack((pnp_points_2d, np.ones(pnp_points_2d.shape[1])))
        x3_x1_for_pnp = x3_for_pnp[:, :x1_for_pnp.shape[1]]
        x3_x1_for_pnp_h = np.vstack((x3_x1_for_pnp, np.ones(x3_x1_for_pnp.shape[1])))
        
        # Select correct initial pose
        # Computing the correct pose with x1_for_pnp_h.shape[1]. In this case we are not filtering, hence not returning X, x1 and x2
        R_c3_c1_initial, t_c3_c1_initial, _, _, _, _ = select_correct_pose_flexible_and_filter(x1_for_pnp_h, x3_x1_for_pnp_h, K, K,
                                                                                             R_c3_c1_option1, R_c3_c1_option2, 
                                                                                         t_c3_c1, plot_FLAG = False,
                                                                                         filtering = False)
        
        if R_c3_c1_initial is None:
            raise ValueError("No valid pose found")
        
        #Convert R matrix to Rvec for initial guess of PnP
        rvec_c3_c1_initial = crossMatrixInv(logm(R_c3_c1_initial.astype('float64')))

        # hstack R and t to initial guess
        pnp_intial_guess = np.hstack((rvec_c3_c1_initial, t_c3_c1_initial))

        # # Solve own PnP
        #pnp_points_3d are in a different to use it later for the cv2 PnP comparison
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
        save_matrix(os.path.join(cache_dir, "T_c1_c3_pnp.txt"), T_c1_c3_pnp)

        # Visualize 3D
        if ALL_VIZ_FLAG:
            visualize_3D_3cameras(T_c1_c2_opt,T_c1_c3_pnp,X_opt_h)

            ### POSSIBLE BUG:WARNING: translation vector in x and y are negative. In own Pnp are positive.
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
        T_c1_c2_opt = load_matrix(os.path.join(cache_dir, "T_c1_c2_opt.txt"))
        R_c3_c1_pnp = load_matrix(os.path.join(cache_dir, "R_c3_c1_pnp.txt"))
        t_c3_c1_pnp = load_matrix(os.path.join(cache_dir, "t_c3_c1_pnp.txt"))
        T_c1_c3_pnp = load_matrix(os.path.join(cache_dir, "T_c1_c3_pnp.txt"))
        x1_for_pnp = load_matrix(os.path.join(cache_dir, "x1.txt"))
        x2_for_pnp = load_matrix(os.path.join(cache_dir, "x2.txt"))
        x3_for_pnp = load_matrix(os.path.join(cache_dir, "x3.txt"))
        rvec_c2_c1_opt = load_matrix(os.path.join(cache_dir, "theta_c2_c1_opt.txt"))
        R_c2_c1_opt = expm(crossMatrix(rvec_c2_c1_opt))
        t_c2_c1_opt = load_matrix(os.path.join(cache_dir, "t_c2_c1_opt.txt"))

        # Triangulate new points for pair c3-c1
        triangulate_new_points_for_pair(
            database_path, adjacency, images_info,
            c_ids[0], c_ids[2], R_c3_c1_pnp, t_c3_c1_pnp, K,
            plot_residuals=ALL_VIZ_FLAG, img1 = images_list[0], img2 = images_list[2])
        

        # Visualize 3D
        all_triangulated_points_3d = get_all_3d_points(database_path)
        all_triangulated_points_3d = all_triangulated_points_3d.T
        if ALL_VIZ_FLAG:
            visualize_3D_3cameras(T_c1_c3_pnp,T_c1_c2_opt,all_triangulated_points_3d)

        
        # Triangulate for c3-c2
        triangulate_new_points_for_pair_in_c1(database_path, adjacency, images_info,
                                          c_ids[1], c_ids[2],
                                          R_c2_c1_opt, t_c2_c1, R_c3_c1_pnp, t_c3_c1,
                                          K, plot_residuals=True, img2=images_list[1], img3=images_list[2])
        
  
        
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
                "rvec": rvec_c3_c1_pnp,
                "tvec": t_c3_c1_pnp,
                "index": 1
            },
        }

        #Plot initial residuals
        visualize_residuals_from_cameras(obs_list, points_3d_dict, camera_data, K, images_list, c_ids)


        
         # Run BA
        result = run_incremental_ba(camera_data, obs_list, points_3d_dict, K)

        ba_cuhunks_dir = "ba_chunks"
        ba_cachedir = os.path.join(cache_dir, ba_cuhunks_dir)
        np.save(os.path.join(ba_cachedir, "ba_params_full_least_production.npy"), result)

    
    if VISUALIZATION_FULL_BUNDLE_3CAMERAS_FLAG:
        # Load data
        # load from cache
        ba_cuhunks_dir = "ba_chunks"
        ba_cachedir = os.path.join(cache_dir, ba_cuhunks_dir)
        params_opt = np.load(os.path.join(ba_cachedir, "ba_params_full_least_production.npy"))
        # params_opt = np.load(os.path.join(ba_cachedir, "ba_params_chunk_2.npy"))


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
                "rvec": rvec_c3_c1_pnp,
                "tvec": t_c3_c1_pnp,
                "index": 1
            },
        }

        # calculate the number of free cameras
        n_free = len([cid for cid, cinfo in camera_data.items() if not cinfo["fixed"]])
        offset_3d = 6*n_free
        # WARNING: the 3D points where stacked in another other than the prvious bundle
        X_c1_opt = params_opt[offset_3d:]
        # Recover X_c1_opt that was flatten in X,Y,Z to a 3xN np array
        X_c1_opt = X_c1_opt.reshape(-1, 3)
        X_c1_opt = X_c1_opt.T

        # get unique point ids from points_3d_dict
        unique_pids = list(points_3d_dict.keys())
        
        # update camera_data
        for cid, cinfo in camera_data.items():
            if cinfo["fixed"]:
                # no changes
                continue
            idx = cinfo["index"]
            start = idx*6
            rvec_opt = params_opt[start:start+3]
            tvec_opt = params_opt[start+3:start+6]
            camera_data[cid]["rvec"] = rvec_opt
            camera_data[cid]["tvec"] = tvec_opt
            save_matrix(os.path.join(cache_dir, f"rvec_{cid}_opt.txt"), rvec_opt)
            save_matrix(os.path.join(cache_dir, f"tvec_{cid}_opt.txt"), tvec_opt)
    
        # update points
        for i, pid in enumerate(unique_pids):
            points_3d_dict[pid] = X_c1_opt[:,i]  # store the updated coords
          
        # Extract T matrices to plot camera poses
        T_c1_c3_opt = extract_camera_pose(camera_data, c_ids[2])
        T_c1_c2_opt = extract_camera_pose(camera_data, c_ids[1])

        # Plot camera poses and 3D points optimized
        visualize_3D_3cameras(T_c1_c2_opt,T_c1_c3_opt,X_c1_opt)

        #Plot optimized residuals
        visualize_residuals_from_cameras(obs_list, points_3d_dict, camera_data, K, images_list, c_ids)



        #TODO: update cameras poses and 3D points in the database
        update_3d_points_in_db(database_path, points_3d_dict)
        

    if PNP_CAMERA4_FLAG:
        # load from cache
        rvec_c2_c1 = load_matrix(os.path.join(cache_dir, f"rvec_{c_ids[1]}_opt.txt"))
        tvec_c2_c1 = load_matrix(os.path.join(cache_dir, f"tvec_{c_ids[1]}_opt.txt"))
        rvec_c3_c1 = load_matrix(os.path.join(cache_dir, f"rvec_{c_ids[2]}_opt.txt"))
        tvec_c3_c1 = load_matrix(os.path.join(cache_dir, f"tvec_{c_ids[2]}_opt.txt"))

        # load from db
        X_c1 = get_all_3d_points(database_path)
        X_c1 = X_c1.T

        # Since images_info is empty when the script is run without inserting in db, we need to load the images info fo safety
        # Mainly when debugging
        images_info = rebuild_images_info(database_path, images_info)


        # Add camera 4 to the system and upload with previous data. 
        # TODO: this should be done first in the db as with the 3D points
        camera_data = {
            c_ids[0]: {
                "fixed": True,
                "rvec": np.zeros(3),  
                "tvec": np.zeros(3),
                "index": None         # not used if fixed
            },
            c_ids[1]: {
                "fixed": False,
                "rvec": rvec_c2_c1,  
                "tvec": tvec_c2_c1,
                "index": 0              # offset for indexing
            },
            c_ids[2]: {
                "fixed": False,
                "rvec": rvec_c3_c1,
                "tvec": tvec_c3_c1,
                "index": 1
            },
            c_ids[3]: {
                "fixed": False,
                "rvec": None,
                "tvec": None,
                "index": 2
            },
        }

        # See which 3D points are seen by camera 4
        # TODO: This should be automated with a function. 
        pnp_points_2d = []
        pnp_points_3d = []
        x1_for_pnp = []
        x2_for_pnp = []
        x3_for_pnp = []
        x4_for_pnp = []

        pnp_points_2d, pnp_points_3d, x1_for_pnp = get_points_seen_by_camera(database_path, images_info, c_ids[3],c_ids[0], 
                                                        match_list_4_1, pnp_points_2d, pnp_points_3d, x1_for_pnp)
        pnp_points_2d, pnp_points_3d, x2_for_pnp = get_points_seen_by_camera(database_path, images_info, c_ids[3],c_ids[1],
                                                        match_list_4_2, pnp_points_2d, pnp_points_3d, x2_for_pnp)
        pnp_points_2d, pnp_points_3d, x3_for_pnp = get_points_seen_by_camera(database_path, images_info, c_ids[3],c_ids[2],
                                                        match_list_4_3, pnp_points_2d, pnp_points_3d, x3_for_pnp)
        
        pnp_points_2d = np.array(pnp_points_2d, dtype=np.float64).T    # shape (2, N)
        x1_for_pnp = np.array(x1_for_pnp, dtype=np.float64).T        # shape (2, N)
        x2_for_pnp = np.array(x2_for_pnp, dtype=np.float64).T        # shape (2, N)
        x3_for_pnp = np.array(x3_for_pnp, dtype=np.float64).T        # shape (2, N)
        pnp_points_3d = np.array(pnp_points_3d, dtype=np.float64)
        x4_for_pnp = pnp_points_2d

        debug_print(f"2D points from {image_names[3]} ")
        debug_print(pnp_points_2d.shape)
        debug_print(f"3D points from {image_names[3]} ")
        debug_print(pnp_points_3d.shape)
        debug_print(f"\n")
        

        print(f"New camera sees {x4_for_pnp.shape[1]} 3D points out of {X_c1.shape[1]}")

        # Get initial guess for PnP
        # Extract R and t from F
        R_c4_c1_option1, R_c4_c1_option2, t_c4_c1, F41 = extract_R_t_from_F(database_path, image_names[0], 
                                                                            image_names[3], K)

                                                                        
        # Convert x1_for_pnp  pnp_points_2d to homogeneous coordinates
        x1_for_pnp_h = np.vstack((x1_for_pnp, np.ones(x1_for_pnp.shape[1])))
        x2_for_pnp_h = np.vstack((x2_for_pnp, np.ones(x2_for_pnp.shape[1])))
        x3_for_pnp_h = np.vstack((x3_for_pnp, np.ones(x3_for_pnp.shape[1])))
        x4_for_pnp_h = np.vstack((x4_for_pnp, np.ones(x4_for_pnp.shape[1])))
                                    
        # Select correct initial pose
        # Computing the correct pose with x1_for_pnp_h.shape[1]. In this case we are not filtering, hence not returning X, x1 and x2
        R_c4_c1_initial, t_c4_c1_initial, _, _, _, _ = select_correct_pose_flexible_and_filter(x1_for_pnp_h, x4_for_pnp_h, K, K,
                                                                                                R_c4_c1_option1, R_c4_c1_option2, 
                                                                                                t_c4_c1, plot_FLAG = False,
                                                                                                filtering = False)

        if R_c4_c1_initial is None:
            raise ValueError("No valid pose found")
        
        #Convert R matrix to Rvec for initial guess of PnP
        rvec_c4_c1_initial = crossMatrixInv(logm(R_c4_c1_initial.astype('float64')))
        # hstack R and t to initial guess
        pnp_intial_guess = np.hstack((rvec_c4_c1_initial, t_c4_c1_initial))

        # # Solve own PnP
        rvec_c4_c1_pnp, t_c4_c1_pnp = own_PnP(pnp_points_3d, pnp_points_2d, K, pnp_intial_guess)

        print("Own PnP success:", rvec_c4_c1_pnp, t_c4_c1_pnp)
        save_matrix(os.path.join(cache_dir, "rvec_c4_c1_pnp.txt"), rvec_c4_c1_pnp)
        save_matrix(os.path.join(cache_dir, "t_c4_c1_pnp.txt"), t_c4_c1_pnp)

        #Update camera data
        camera_data[c_ids[3]]["rvec"] = rvec_c4_c1_pnp
        camera_data[c_ids[3]]["tvec"] = t_c4_c1_pnp

        # Recover R matrix from rvec
        R_c4_c1_pnp = expm(crossMatrix(rvec_c4_c1_pnp))

        # Ensemble T matrix
        T_c4_c1_pnp = ensamble_T(R_c4_c1_pnp, t_c4_c1_pnp)

        # Invert T matrix to get T_c1_c4 and plot
        T_c1_c4_pnp = np.linalg.inv(T_c4_c1_pnp)
        save_matrix(os.path.join(cache_dir, "T_c1_c4_pnp.txt"), T_c1_c4_pnp)

        # Recover T_c1_c2 and T_c1_c3 form rvec and tvec
        T_c1_c2 = extract_camera_pose(camera_data, c_ids[1])
        T_c1_c3 = extract_camera_pose(camera_data, c_ids[2])
        T_c1_c4 = extract_camera_pose(camera_data, c_ids[3])

        # Visualize 3D
        # TODO: automate function to plot N cameras
        if ALL_VIZ_FLAG:
            visualize_3D_4cameras(T_c1_c2,T_c1_c3,T_c1_c4_pnp,X_c1)

    if TRIANGULATE_CAMERA_4_FLAG:

        R_c4_c1 = expm(crossMatrix(rvec_c4_c1_pnp))
        R_c2_c1 = expm(crossMatrix(rvec_c2_c1))
        R_c3_c1 = expm(crossMatrix(rvec_c3_c1))

        
        # Triangulate new points for pair c4-c1
        # BUG: plotting without c4. Maybe double plot
        triangulate_new_points_for_pair(database_path, adjacency, images_info,
                                        c_ids[0], c_ids[3],
                                        R_c4_c1, t_c4_c1_pnp, K, plot_residuals=True, 
                                        img1 = images_list[0], img2 = images_list[3])


        # Visualize 3D
        all_triangulated_points_3d = get_all_3d_points(database_path)
        all_triangulated_points_3d = all_triangulated_points_3d.T
        if ALL_VIZ_FLAG:
            visualize_3D_4cameras(T_c1_c2,T_c1_c3,T_c1_c4_pnp,all_triangulated_points_3d)

        # Recover ts from camera data
        t_c2_c1 = camera_data[c_ids[1]]["tvec"]
        t_c3_c1 = camera_data[c_ids[2]]["tvec"]

        
        # Triangulate for c4-c2
        # BUG: plotting without c4. Maybe double plot
        triangulate_new_points_for_pair_in_c1(database_path, adjacency, images_info,
                                            c_ids[1], c_ids[3],
                                            R_c2_c1, t_c2_c1, R_c4_c1, t_c4_c1_pnp,
                                            K, plot_residuals=True, img2=images_list[1], img3=images_list[3])
        
        # Visualize 3D
        all_triangulated_points_3d = get_all_3d_points(database_path)
        all_triangulated_points_3d = all_triangulated_points_3d.T
        if ALL_VIZ_FLAG:
            visualize_3D_4cameras(T_c1_c2,T_c1_c3,T_c1_c4_pnp,all_triangulated_points_3d)

        #Triangulate for c4-c3
        # BUG: plotting without c4. Maybe double plot
        triangulate_new_points_for_pair_in_c1(database_path, adjacency, images_info,
                                            c_ids[2], c_ids[3],
                                            R_c3_c1, t_c3_c1, R_c4_c1, t_c4_c1_pnp,
                                            K, plot_residuals=True, img2=images_list[2], img3=images_list[3])

        # Visualize 3D
        all_triangulated_points_3d = get_all_3d_points(database_path)
        all_triangulated_points_3d = all_triangulated_points_3d.T
        if ALL_VIZ_FLAG:
            visualize_3D_4cameras(T_c1_c2,T_c1_c3,T_c1_c4_pnp,all_triangulated_points_3d)
    
    if FULL_BUNDLE_4CAMERAS_FLAG:
                
        obs_list, points_3d_dict = load_observations_and_points(database_path)

        #Plot initial residuals
        visualize_residuals_from_cameras(obs_list, points_3d_dict, camera_data, K, images_list, c_ids)

        # Run BA
        result = run_incremental_ba(camera_data, obs_list, points_3d_dict, K)

        ba_cuhunks_dir = "ba_chunks"
        ba_cachedir = os.path.join(cache_dir, ba_cuhunks_dir)
        np.save(os.path.join(ba_cachedir, "ba_params_4_cameras.npy"), result)
    
    if FULL_BUNDLE_4CAMERAS_VISUALIZATION_FLAG:
        # Load data
        # load from cache
        ba_cuhunks_dir = "ba_chunks"
        ba_cachedir = os.path.join(cache_dir, ba_cuhunks_dir)
        params_opt = np.load(os.path.join(ba_cachedir, "ba_params_4_cameras.npy"))

        obs_list, points_3d_dict = load_observations_and_points(database_path)
        
        camera_data = {
            c_ids[0]: {
                "fixed": True,
                "rvec": np.zeros(3),  
                "tvec": np.zeros(3),
                "index": None         # not used if fixed
            },
            c_ids[1]: {
                "fixed": False,
                "rvec": None,  
                "tvec": None,
                "index": 0              # offset for indexing
            },
            c_ids[2]: {
                "fixed": False,
                "rvec": None,
                "tvec": None,
                "index": 1
            },
            c_ids[3]: {
                "fixed": False,
                "rvec": None,
                "tvec": None,
                "index": 2
            },
        }

        # calculate the number of free cameras
        n_free = len([cid for cid, cinfo in camera_data.items() if not cinfo["fixed"]])
        offset_3d = 6*n_free
        # WARNING: the 3D points where stacked in another other than the prvious bundle
        X_c1_opt = params_opt[offset_3d:]
        # Recover X_c1_opt that was flatten in X,Y,Z to a 3xN np array
        X_c1_opt = X_c1_opt.reshape(-1, 3)
        X_c1_opt = X_c1_opt.T

        # get unique point ids from points_3d_dict
        unique_pids = list(points_3d_dict.keys())
        
        # update camera_data
        for cid, cinfo in camera_data.items():
            if cinfo["fixed"]:
                # no changes
                continue
            idx = cinfo["index"]
            start = idx*6
            rvec_opt = params_opt[start:start+3]
            tvec_opt = params_opt[start+3:start+6]
            camera_data[cid]["rvec"] = rvec_opt
            camera_data[cid]["tvec"] = tvec_opt
            save_matrix(os.path.join(cache_dir, f"rvec_{cid}_opt_2.txt"), rvec_opt)
            save_matrix(os.path.join(cache_dir, f"tvec_{cid}_opt_2.txt"), tvec_opt)
        
        # update points
        for i, pid in enumerate(unique_pids):
            points_3d_dict[pid] = X_c1_opt[:,i]  # store the updated coords
        
        # Extract T matrices to plot camera poses
        
        T_c1_c2_opt = extract_camera_pose(camera_data, c_ids[1])
        T_c1_c3_opt = extract_camera_pose(camera_data, c_ids[2])
        T_c1_c4_opt = extract_camera_pose(camera_data, c_ids[3])

        # Plot camera poses and 3D points optimized
        visualize_3D_4cameras(T_c1_c2_opt,T_c1_c3_opt,T_c1_c4_opt,X_c1_opt)

        #Plot optimized residuals
        visualize_residuals_from_cameras(obs_list, points_3d_dict, camera_data, K, images_list, c_ids)






#Step 8 Repeat steps 4 to 8 for the rest of the cameras. Some effort should be done to automate the process


