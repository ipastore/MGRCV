import numpy as np
import cv2
import matplotlib.pyplot as plt

# Define the distortion correction
def undistort_kannala_brandt(image, fx, fy, cx, cy, k1, k2, k3, k4):
    h, w = image.shape[:2]
    map_x, map_y = np.meshgrid(np.arange(w), np.arange(h))
    map_x = (map_x - cx) / fx
    map_y = (map_y - cy) / fy
    r = np.sqrt(map_x**2 + map_y**2)
    theta = np.arctan(r)

    # Apply Kannala-Brandt model (simplified approximation)
    theta_d = theta + k1 * theta**3 + k2 * theta**5 + k3 * theta**7 + k4 * theta**9
    scale = np.tan(theta_d) / r
    map_x = map_x * scale * fx + cx
    map_y = map_y * scale * fy + cy

    # Remap the image
    undistorted_image = cv2.remap(image, map_x.astype(np.float32), map_y.astype(np.float32), interpolation=cv2.INTER_LINEAR)
    return undistorted_image

# Apply the function
distorted = cv2.imread("/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/data/Seq_035/f_0082928.png")
# distorted = cv2.imread("/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/data/Seq_035/f_0082412.png")

fx, fy, cx, cy = 727.1851, 728.5954, 668.1817, 507.4003
k1, k2, k3, k4 = -0.1311029, -0.005149247, 0.001512357, -6.998448e-05
undistorted_own = undistort_kannala_brandt(distorted, fx, fy, cx, cy, k1, k2, k3, k4)

################ OpenCV Implementation ################

# Define camera matrix and distortion coefficients as numpy arrays
K = np.array([[fx, 0, cx],
              [0, fy, cy],
              [0, 0, 1]], dtype=np.float32)

D = np.array([[k1, k2, k3, k4]], dtype=np.float32)
h, w = 1012, 1350  # Assuming original image dimensions
DIM = (w, h)
dim2 = None
dim3 = None
balance = 0.0


dim1 = distorted.shape[:2][::-1]  #dim1 is the dimension of input image to un-distort
    
assert dim1[0]/dim1[1] == DIM[0]/DIM[1], "Image to undistort needs to have same aspect ratio as the ones used in calibration"

if not dim2:
    dim2 = dim1
if not dim3:
    dim3 = dim1

scaled_K = K * dim1[0] / DIM[0]  # The values of K is to scale with image dimension.

scaled_K[2][2] = 1.0  # Except that K[2][2] is always 1.0

# This is how scaled_K, dim2 and balance are used to determine the final K used to un-distort image. OpenCV document failed to make this clear!
new_K = cv2.fisheye.estimateNewCameraMatrixForUndistortRectify(scaled_K, D, dim2, np.eye(3), balance=balance)

undistorted_open_cv = cv2.fisheye.undistortImage(distorted, scaled_K, D, Knew=new_K, new_size=dim3)


############ OPEN CV ONE LINE IMPLEMENTATION ################
# Undistort the image using the camera matrix and distortion coefficients
undistorted_open_cv_oneline = cv2.undistort(distorted, K, D)


# Display the images side by side
fig, ax = plt.subplots(1, 4, figsize=(12, 6))
ax[0].imshow(cv2.cvtColor(distorted, cv2.COLOR_BGR2RGB))
ax[0].set_title("Distorted Image")
ax[0].axis("off")
ax[1].imshow(cv2.cvtColor(undistorted_own, cv2.COLOR_BGR2RGB))
ax[1].set_title("Undistorted Image OWN")
ax[1].axis("off")
ax[2].imshow(cv2.cvtColor(undistorted_own, cv2.COLOR_BGR2RGB))
ax[2].set_title("Undistorted Image OPEN CV")
ax[2].axis("off")
ax[3].imshow(cv2.cvtColor(undistorted_open_cv_oneline, cv2.COLOR_BGR2RGB))
ax[3].set_title("Undistorted Image OPEN CV ONE LINE")
ax[3].axis("off")
plt.show()



