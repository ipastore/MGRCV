def resBundleProjection(Op, x_data, T_wc1, T_wc2, K_1, K_2, D1_k_array, D2_k_array, nPairs=2):
    """
    Compute residuals between observed 2D points and projected 3D points.
    
    Parameters:
        Op: Optimization parameters (rotation, translation, 3D points).
        x_data: Observed 2D points (array of shape 2x[nPairs * nPoints]).
        T_wc1: Pose of camera 1 in world coordinates (4x4).
        T_wc2: Pose of camera 2 in world coordinates (4x4).
        K_1: Camera 1 intrinsic calibration matrix (3x3).
        K_2: Camera 2 intrinsic calibration matrix (3x3).
        D1_k_array: Distortion coefficients for camera 1.
        D2_k_array: Distortion coefficients for camera 2.
        nPairs: Number of camera pairs.
        
    Returns:
        res: Residuals (difference between observed and reprojected points).
    """

    def project_points_to_cameras_kannala(X_3D, T_c1w, T_c2w, K_1, K_2, D1, D2):
        """Project 3D points onto the two cameras and return the projections."""

        # Transform 3D points to each camera's frame
        x_3d_c1 = T_c1w @ X_3D
        x_3d_c2 = T_c2w @ X_3D

        # Project using the Kannala-Brandt model
        u_1_array = kannala_forward_model(x_3d_c1, K_1, D1)
        u_2_array = kannala_forward_model(x_3d_c2, K_2, D2)

        return u_1_array, u_2_array

    def compute_residuals(x_observed, u_projected, nPoints):
        """Compute residuals between observed and projected points."""
        res = []
        for i in range(nPoints):
            res.append(x_observed[0, i] - u_projected[0, i])
            res.append(x_observed[1, i] - u_projected[1, i])
        return res

    # --- Extract Parameters ---
    posStartX = 6 * (nPairs - 1)  # 6 parameters (rotation + translation) per pair (except the first)
    nPoints = int((Op.shape[0] - posStartX) / 3)  # Number of 3D points
    X_3D = np.vstack([Op[posStartX:].reshape(3, nPoints), np.ones((1, nPoints))])  # Convert to homogeneous coordinates

    # --- Initial Projections for Camera Pair 1 ---
    T_c1w = np.linalg.inv(T_wc1)  # Camera 1 relative to world
    T_c2w = np.linalg.inv(T_wc2)  # Camera 2 relative to world
    u_1_array, u_2_array = project_points_to_cameras(X_3D, T_c1w, T_c2w, K_1, K_2, D1_k_array, D2_k_array)

    # --- Residuals for Camera Pair 1 ---
    res = compute_residuals(x_data[:, :nPoints], u_1_array, nPoints)
    res += compute_residuals(x_data[:, nPoints:2 * nPoints], u_2_array, nPoints)

    # --- Residuals for Additional Pairs (nPairs > 1) ---
    for i in range(nPairs - 1):
        theta_rot = Op[i * 6:i * 6 + 3]
        t_theta = Op[i * 6 + 3]
        t_phi = Op[i * 6 + 4]
        tras = np.array([
            np.sin(t_theta) * np.cos(t_phi),
            np.sin(t_theta) * np.sin(t_phi),
            np.cos(t_theta),
        ])

        # Construct the transformation for pair i
        T_wAwB = ObtainPose(theta_rot, t_theta, t_phi)
        T_wAwB[0:3, 3] = tras

        for j in range(nPoints):
            # Transform 3D points to the additional camera's frame
            x_3d_B = T_wAwB @ X_3D[:, j]
            x_3d_1B = T_c1w @ x_3d_B
            x_3d_2B = T_c2w @ x_3d_B

            # Project the transformed points
            u_1_B = kannala_forward_model(x_3d_1B, K_1, D1_k_array)
            u_2_B = kannala_forward_model(x_3d_2B, K_2, D2_k_array)

            # Compute residuals for the additional pairs
            res.append(x_data[0, j + 2 * i * nPoints + 2 * nPoints] - u_1_B[0])
            res.append(x_data[1, j + 2 * i * nPoints + 2 * nPoints] - u_1_B[1])
            res.append(x_data[0, j + 2 * i * nPoints + 2 * nPoints + nPoints] - u_2_B[0])
            res.append(x_data[1, j + 2 * i * nPoints + 2 * nPoints + nPoints] - u_2_B[1])

    return res
