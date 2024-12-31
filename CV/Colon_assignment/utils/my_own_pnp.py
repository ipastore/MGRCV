import numpy as np
from scipy.optimize import least_squares

def pnp(pts_3D, pts_2D, K):
    """
    Simple PnP implementation using least squares optimization.
    :param pts_3D: Nx3 array of 3D points.
    :param pts_2D: Nx2 array of 2D image points.
    :param K: 3x3 camera intrinsic matrix.
    :return: Rotation matrix (3x3) and translation vector (3x1).
    """

    # Initial guess: no rotation, translation based on the centroid of 3D points
    # TODO: initial guess as extracted from Fundamental matrix
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
    R = rodriguesToRmat(R_vec)
    projected_pts = (K @ (R @ pts_3D.T + t.reshape(-1, 1))).T
    projected_pts /= projected_pts[:, 2].reshape(-1, 1)  # Normalize
    residuals = (projected_pts[:, :2] - pts_2D).ravel()
    
    return residuals