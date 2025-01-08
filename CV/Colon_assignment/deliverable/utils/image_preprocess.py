import cv2
import os
import matplotlib.pyplot as plt
import yaml
import numpy as np

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


seq_name = "Seq_027"
seq_path = f"../data/{seq_name}"
output_dir = f"../data/{seq_name}/resized_undistorted/"
calibration_path = f"../data/calib.yaml"
os.makedirs(output_dir, exist_ok=True)

#read YAML file: 
with open(calibration_path,"r") as file:
    calib = yaml.load(file, Loader=yaml.FullLoader)
    print(calib)


# Extract the camera matrix
fx = calib["Camera"]["fx"]
fy = calib["Camera"]["fy"]
cx = calib["Camera"]["cx"]
cy = calib["Camera"]["cy"]


# Extract the distortion coefficients
k1 = calib["Camera"]["k1"]
k2 = calib["Camera"]["k2"]
k3 = calib["Camera"]["k3"]
k4 = calib["Camera"]["k4"]


for image in os.listdir(seq_path):
    if not image.endswith(".png"):
        continue

    #Load image
    image_path = os.path.join(seq_path, image)
    image_cv = cv2.imread(image_path)

    # Undistort image
    image_undistorted = undistort_kannala_brandt(image_cv, fx, fy, cx, cy, k1, k2, k3, k4)

    # Resize image
    image_resized = cv2.resize(image_undistorted, (960, 720))

    # # Display images side by side
    # fig, ax = plt.subplots(1, 2, figsize=(12, 6))
    # ax[0].imshow(cv2.cvtColor(image_cv, cv2.COLOR_BGR2RGB))
    # ax[0].set_title("Original Image")
    # ax[0].axis("off")
    # ax[1].imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
    # ax[1].set_title("Resized and Undistorted Image")
    # ax[1].axis("off")
    # plt.show()

    # Save image
    output_path = os.path.join(output_dir, image)
    cv2.imwrite(output_path, image_resized)
    print(f"Resized {output_path}")

