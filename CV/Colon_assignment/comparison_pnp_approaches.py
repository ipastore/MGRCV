import numpy as np
import os
import cv2
from scipy.optimize import least_squares



def solvePnP(objectPoints, imagePoints, cameraMatrix, distCoeffs=None, 
             flags="SOLVEPNP_EPNP", max_iter=100, eps=1e-6):
    """
    A *conceptual* Python function for EPNP, mimicking cv2.solvePnP signature.
    
    Parameters
    ----------
    objectPoints : np.ndarray
        3D points in the object reference frame of shape (N, 3).
    imagePoints : np.ndarray
        Corresponding 2D points in the image plane of shape (N, 2).
    cameraMatrix : np.ndarray
        The 3x3 camera intrinsic matrix.
    distCoeffs : np.ndarray or None
        Distortion coefficients, shape can be (4,) or (5,) or more, depending on model.
        For simplicity, we assume we either have None or we do not handle them in detail.
    flags : str
        String specifying the method. Here we only illustrate "SOLVEPNP_EPNP".
    max_iter : int
        Maximum number of iterations for any optional refinement.
    eps : float
        Convergence threshold for optional refinement.

    Returns
    -------
    retval : bool
        True if success, False otherwise.
    rvec : np.ndarray
        Rotation vector (Rodrigues form) of shape (3,).
    tvec : np.ndarray
        Translation vector of shape (3,).
    """
    if flags != "SOLVEPNP_EPNP":
        raise NotImplementedError("Only EPNP concept is demonstrated here.")

    # ------------------------------------------------------------------
    # 0. Preprocessing: (Optional) Undistort points if distCoeffs is given
    # ------------------------------------------------------------------
    # In a real system, you might undistort imagePoints here using the known
    # distortion parameters. For simplicity, we skip it or assume it's done already.

    # ------------------------------------------------------------------
    # 1. Check data shapes
    # ------------------------------------------------------------------
    assert len(objectPoints) == len(imagePoints), \
        "objectPoints and imagePoints must have the same length."
    N = len(objectPoints)
    if N < 4:
        raise ValueError("EPNP requires at least 4 point correspondences.")

    # Convert inputs to float64 numpy arrays if they aren’t already
    objectPoints = np.ascontiguousarray(objectPoints, dtype=np.float64)
    imagePoints = np.ascontiguousarray(imagePoints, dtype=np.float64)
    cameraMatrix = np.ascontiguousarray(cameraMatrix, dtype=np.float64)

    # ------------------------------------------------------------------
    # 2. (Conceptual) Build the 4 virtual control points in object space
    # ------------------------------------------------------------------
    # A typical approach chooses 3 or 4 reference points (control points) from the data.
    # For demonstration, we pick 4 points and treat them as "control points."
    # More sophisticated methods do a PCA or more advanced selection.
    #
    # We'll keep it simple and just pick an arbitrary subset or do a simple approach.
    cp_idx = [0, 1, 2, 3]  # In reality, you'd have a more robust selection here.
    control_points_3D = objectPoints[cp_idx, :]  # shape (4, 3)

    # ------------------------------------------------------------------
    # 3. Express each 3D object point as a barycentric combination of the control points
    # ------------------------------------------------------------------
    # In EPNP, you solve for α_{i,j} (barycentric coords) s.t.
    #   X_i = \sum_{j=1 to 4} α_{i,j} * P_j
    # with the constraint \sum_{j=1 to 4} α_{i,j} = 1, and α_{i,j} >= 0.
    #
    # For simplicity, we'll do a naive linear system approach or direct solve.
    # In a real EPNP, you might do a more robust least-squares approach.

    # Build matrix M such that M * alpha_i = X_i,
    # but we also need the constraint sum(alpha_i)=1 for each X_i.
    # Let's do a simplified approach or direct solver for illustration:
    
    # We'll solve separately for each point's alpha. This can be more elaborate in real EPNP.
    # control_points_3D is 4x3, we want alpha_i (4,) for each X_i.
    # Because we also have the constraint sum(alpha_i) = 1, we do a small workaround.

    def get_alphas(X, C):
        # Solve for alpha in X = sum_j alpha_j * C_j subject to sum_j alpha_j = 1
        # We can rewrite: X = A alpha, with A=[C_1^T - C_4^T, ...], or do a direct approach.
        # For demonstration, a naive approach:
        #   X = C^T * alpha, sum(alpha)=1 => we can add an extra row for that constraint
        # i.e. M is 4x4, last row is [1,1,1,1], and b is [X, 1].
        M = np.vstack([C.T, np.ones((1,4))])    # shape (4, 4)
        b = np.append(X, 1).reshape(-1,1)       # shape (4, 1)
        alpha, residuals, rank, s = np.linalg.lstsq(M, b, rcond=None)
        return alpha.ravel()  # shape (4,)

    C = control_points_3D  # shape (4,3)
    alphas_list = []
    for i in range(N):
        X_i = objectPoints[i]
        a_i = get_alphas(X_i, C)
        alphas_list.append(a_i)
    alphas = np.array(alphas_list)  # shape (N,4)

    # ------------------------------------------------------------------
    # 4. Build the 2D constraints from imagePoints and cameraMatrix.
    # ------------------------------------------------------------------
    # Let the "control points in camera frame" be unknown (4,3).
    # We call them Pc_j = (Xc_j, Yc_j, Zc_j), j=1..4.
    #
    # Then the projection of X_i in the image = 
    #    K * [ \sum_j alpha_{i,j} Pc_j ] / (Z_i in camera)
    #
    # We want to solve for the 12 unknowns: Pc_1, Pc_2, Pc_3, Pc_4 (each has 3 coords).
    # That’s 12 unknowns, but the system can be quite large. We do a linear approach for each
    # 2D point. Then we typically get multiple candidate solutions. We keep the physically valid one.
    #
    # Real EPNP does a polynomial solving approach; we’ll do a naive linear system here.

    # Let's build a large matrix A for the linear system. We'll do a simplified approach
    # ignoring the fact that real EPNP uses a more specialized polynomial solver.

    # If Pi_c = (Xcj, Ycj, Zcj), then for point i:
    #
    #   imagePoints[i, 0] ~ fx * Xi_c / Zi_c + cx
    #   imagePoints[i, 1] ~ fy * Yi_c / Zi_c + cy
    #
    # where Xi_c, Yi_c, Zi_c = sum_j (alpha_{i,j} * Xc_j, alpha_{i,j} * Yc_j, alpha_{i,j} * Zc_j).
    #
    # We can rewrite each projection equation in a linearized (approx) form or use a minimal parameterization. 
    # For demonstration, let's just show the placeholders. A real solution is more intricate.

    # We'll do a simplified 3-step approach:
    # 1) Guess initial location for the 4 control points in front of the camera (Z > 0).
    # 2) Refine by iterating a small Levenberg–Marquardt on the full reprojection error.
    # 3) Extract R, t from the final 3D control points.

    # Step 4.1: Provide an initial guess: place the control points in front of camera
    Pc_init = np.copy(control_points_3D)
    # For instance, assume the object is near the front and apply a random scale/translation
    Pc_init[:, 2] += 5.0  # push them a bit in front of the camera
    Pc_est = Pc_init.ravel()  # shape (12,)

    # Step 4.2: Define a function to compute reprojection error given the Pc_est
    def reprojection_error(Pc_est_vec):
        Pc_est_mat = Pc_est_vec.reshape((4,3))  # 4 control points in camera frame
        fx = cameraMatrix[0, 0]
        fy = cameraMatrix[1, 1]
        cx = cameraMatrix[0, 2]
        cy = cameraMatrix[1, 2]

        errors = []
        for i in range(N):
            # Compute Xi_c, Yi_c, Zi_c
            Xi_c = np.sum(alphas[i] * Pc_est_mat[:, 0])
            Yi_c = np.sum(alphas[i] * Pc_est_mat[:, 1])
            Zi_c = np.sum(alphas[i] * Pc_est_mat[:, 2])

            if Zi_c <= 1e-8:
                # If we get a negative or zero depth, penalize heavily
                errors.append(1e3)
                errors.append(1e3)
                continue

            # Project
            u_proj = fx*(Xi_c/Zi_c) + cx
            v_proj = fy*(Yi_c/Zi_c) + cy

            # Compare with the measured 2D
            u_meas, v_meas = imagePoints[i]
            errors.append(u_meas - u_proj)
            errors.append(v_meas - v_proj)

        return np.array(errors, dtype=np.float64)

    # Step 4.3: Levenberg–Marquardt (very naive)
    # We'll do a small numeric Jacobian + iterative update:
    def gauss_newton_lm(Pc_init, max_iter=50, lambd=1e-3):
        current = Pc_init
        for it in range(max_iter):
            err = reprojection_error(current)
            cost = 0.5 * np.dot(err, err)
            if cost < eps:
                break

            # Numeric Jacobian
            J = []
            eps_jac = 1e-6
            for idx in range(len(current)):
                delta = np.zeros_like(current)
                delta[idx] = eps_jac

                err_p = reprojection_error(current + delta)
                err_m = reprojection_error(current - delta)
                Jcol = (err_p - err_m)/(2.0*eps_jac)
                J.append(Jcol)
            J = np.array(J).T  # shape (2N, 12)

            # Gauss-Newton update: (J^T J + λI) \ (J^T * err)
            A = J.T @ J
            np.fill_diagonal(A, A.diagonal() + lambd)
            g = J.T @ err
            step = np.linalg.lstsq(A, g, rcond=None)[0]
            
            # Simple approach: update
            new = current - step

            # Evaluate new cost
            err_new = reprojection_error(new)
            cost_new = 0.5 * np.dot(err_new, err_new)

            if cost_new < cost:
                # Accept the update
                current = new
                # Optionally decrease lambda
                lambd *= 0.5
            else:
                # Reject update, increase lambda
                lambd *= 5.0

            if np.abs(cost - cost_new) < eps:
                break

        return current

    Pc_solution = gauss_newton_lm(Pc_est, max_iter=max_iter)
    # Now we have an estimate for the 4 control points in the camera frame.

    # ------------------------------------------------------------------
    # 5. Recover R, t from the transformation between the object-space control points and camera-space ones.
    # ------------------------------------------------------------------
    # We have:
    #   control_points_3D in object space  ->  Pc_solution in camera space
    # We want the 3D transformation that maps control_points_3D to Pc_solution.
    # That transformation is our extrinsic matrix [R|t], *assuming the same correspondences.
    # We can do a Kabsch / Procrustes approach to find the best rotation and translation.

    Pc_cam = Pc_solution.reshape((4,3))  # shape (4,3)
    # Center them (remove centroids)
    centroid_obj = np.mean(control_points_3D, axis=0)
    centroid_cam = np.mean(Pc_cam, axis=0)
    X_obj_cent = control_points_3D - centroid_obj  # shape (4,3)
    X_cam_cent = Pc_cam - centroid_cam            # shape (4,3)

    # Compute SVD for rotation
    H = X_obj_cent.T @ X_cam_cent  # shape (3,3)
    U, S, Vt = np.linalg.svd(H)
    R_est = Vt.T @ U.T
    # Handle improper rotation (if det(R) < 0)
    if np.linalg.det(R_est) < 0:
        Vt[2,:] *= -1
        R_est = Vt.T @ U.T

    # Translation
    t_est = centroid_cam - R_est @ centroid_obj

    # Convert rotation matrix to Rodrigues vector
    # Rodrigues formula: rvec = angle * axis
    # We can use a small helper:
    def rotationMatrixToRodrigues(R):
        # A simple approach, see also cv2.Rodrigues in practice
        # We find the axis from the skew-symmetric part, angle from trace
        trace = np.trace(R)
        theta = np.arccos((trace - 1.0)/2.0)
        if abs(theta) < 1e-12:
            # No rotation
            return np.zeros((3,))
        rx = R[2,1] - R[1,2]
        ry = R[0,2] - R[2,0]
        rz = R[1,0] - R[0,1]
        r = np.array([rx, ry, rz], dtype=np.float64)
        norm_r = np.linalg.norm(r)
        r /= norm_r
        rvec = r * theta
        return rvec

    rvec_est = rotationMatrixToRodrigues(R_est)
    tvec_est = t_est

    # We can define success if we have a positive average depth for the 4 control points
    # or if the final cost is below some threshold:
    final_err = reprojection_error(Pc_solution)
    final_cost = 0.5 * np.dot(final_err, final_err)
    retval = True
    if np.isnan(final_cost) or final_cost > 1e6:
        retval = False

    return retval, rvec_est, tvec_est


def pnp(pts_3D, pts_2D, K):
    """
    Simple PnP implementation using least squares optimization.
    :param pts_3D: Nx3 array of 3D points.
    :param pts_2D: Nx2 array of 2D image points.
    :param K: 3x3 camera intrinsic matrix.
    :return: Rotation matrix (3x3) and translation vector (3x1).
    """


    # Initial guess: no rotation, translation based on the centroid of 3D points
    centroid_3D = np.mean(pts_3D, axis=0)
    initial_guess = np.zeros(6)
    initial_guess[3:] = centroid_3D - (centroid_3D/2)  # Use the negative centroid as an initial guess for translation

    result = least_squares(reprojection_error, initial_guess, args=(pts_3D, pts_2D, K))
    
    R_vec, t = result.x[:3], result.x[3:]
    # R, _ = cv2.Rodrigues(R_vec)
    return R_vec, t

def rodriguesToRmat(rvec):
    # simplistic version
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])
    R = np.eye(3) + np.sin(theta)*K + (1-np.cos(theta))*(K@K)
    return R

def reprojection_error(params, pts_3D, pts_2D, K):
    R_vec, t = params[:3], params[3:]
    # R, _ = cv2.Rodrigues(R_vec)  # Convert rotation vector to matrix
    R = rodriguesToRmat(R_vec)
    projected_pts = (K @ (R @ pts_3D.T + t.reshape(-1, 1))).T
    projected_pts /= projected_pts[:, 2].reshape(-1, 1)  # Normalize
    residuals = (projected_pts[:, :2] - pts_2D).ravel()
    
    # Debugging: Check for NaN or infinite values
    if not np.all(np.isfinite(residuals)):
        print("Non-finite residuals detected")
        print("R_vec:", R_vec)
        print("t:", t)
        print("projected_pts:", projected_pts)
        print("residuals:", residuals)
    
    return residuals



# ---------------------------
# Example usage (toy example)
# ---------------------------
if __name__ == "__main__":
    # Fake data
    obj_pts = np.array([
        [0., 0., 0.],
        [1., 0., 0.],
        [0., 1., 0.],
        [1., 1., 0.],
        [0.5, 0.5, 2.0],
        [2.0, 2.0, 1.0]
    ])

    # Suppose our camera is looking along -Z, with some rotation/translation.
    # Let's define a ground truth R,t, then project to get image points.

    # Suppose ground truth rvec, tvec
    gt_rvec = np.array([0.2, 0.1, -0.3])  # just some rotation
    R_gt = rodriguesToRmat(gt_rvec)
    t_gt = np.array([0.5, 0.1, 5.0])

    print("Ground truth R =", gt_rvec)
    print("Ground truth t =", t_gt)


    # Camera intrinsics
    fx, fy = 800, 800
    cx, cy = 320, 240
    K_c = np.array([[fx,   0, cx],
                    [ 0,  fy, cy],
                    [ 0,   0,  1]], dtype=np.float64)

    def project_points(pts_3d, R, t, K):
        proj_pts = []
        for X in pts_3d:
            Xc = R @ X + t
            u = K[0,0]*(Xc[0]/Xc[2]) + K[0,2]
            v = K[1,1]*(Xc[1]/Xc[2]) + K[1,2]
            proj_pts.append([u,v])
        return np.array(proj_pts)

    img_pts = project_points(obj_pts, R_gt, t_gt, K_c)

    # Now call our solvePnP
    retval, rvec_est, tvec_est = solvePnP(obj_pts, img_pts, K_c, None, 
                                          flags="SOLVEPNP_EPNP",
                                          max_iter=100, eps=1e-10)
    
    print("retval =", retval)
    print("Estimated rvec =", rvec_est)
    print("Estimated tvec =", tvec_est)

    ###### OPEN CV ########
    # Now let's compare with OpenCV's solvePnP
    retval, rvec_est, tvec_est = cv2.solvePnP(obj_pts, img_pts, K_c, None, flags=cv2.SOLVEPNP_EPNP)

    print("retval (OpenCV) =", retval)
    print("Estimated rvec (OpenCV) =", rvec_est.ravel())
    print("Estimated tvec (OpenCV) =", tvec_est.ravel())

    ###### NAIV pnp implementation ######
    # Now let's compare with our naive pnp implementation
    R_est, t_est = pnp(obj_pts, img_pts, K_c)

    print("Estimated R (naive) =", R_est)
    print("Estimated t (naive) =", t_est)

