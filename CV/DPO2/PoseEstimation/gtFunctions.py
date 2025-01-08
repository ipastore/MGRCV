import numpy as np
from scipy.spatial.transform import Rotation as R

def quaternion_to_rotation_matrix(qw, qx, qy, qz):
    """
    Convert quaternion to rotation matrix.
    
    -input:
        qw: float
            Scalar component of the quaternion.
        qx, qy, qz: float
            Vector components of the quaternion.

    -output:
        r: ndarray (3x3)
            Rotation matrix corresponding to the input quaternion.
    """
    r = np.array([
        [1 - 2 * (qy**2 + qz**2), 2 * (qx*qy - qz*qw), 2 * (qx*qz + qy*qw)],
        [2 * (qx*qy + qz*qw), 1 - 2 * (qx**2 + qz**2), 2 * (qy*qz - qx*qw)],
        [2 * (qx*qz - qy*qw), 2 * (qy*qz + qx*qw), 1 - 2 * (qx**2 + qy**2)]
    ])
    return r


def changePoseTransformation(R_wc, t_wc):

    R_cw = R_wc.T
    t_cw = -R_cw @ t_wc

    return R_cw, t_cw



def extract_camera_poses(images_file):
    """
    Extract camera poses from a COLMAP-style images.txt file.
    
    -input:
        images_file: str
            Path to the images.txt file containing the camera poses.

    -output:
        camera_poses: dict
            Dictionary mapping image names to their corresponding poses.
            Each pose contains:
                - image_id: int
                    Unique identifier of the image.
                - rotation_matrix: ndarray (3x3)
                    Rotation matrix representing the camera orientation.
                - translation_vector: ndarray (3,)
                    Translation vector representing the camera position.
    """
    camera_poses = {}
    
    with open(images_file, 'r') as file:
        lines = file.readlines()
    
    for line in lines:
        if line.startswith("#") or len(line.strip()) == 0:
            # Skip comments and empty lines
            continue
        
        parts = line.strip().split()
        if len(parts) != 10 :
            # Skip the second line of the entry (2D-3D correspondences)
            continue
        
        # Extract pose data
        image_id = int(parts[0])
        qw, qx, qy, qz = map(float, parts[1:5])
        tx, ty, tz = map(float, parts[5:8])
        image_name = parts[9]
        
        # Compute rotation matrix
        rotation_matrix = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        # rotation_mat = R.from_quat([qx, qy, qz, qw]).as_matrix()
        # R_mat = rotation_mat.as_matrix()
        translation_vector = np.array([tx, ty, tz])

        # Store pose
        camera_poses[image_name] = {
            "image_id": image_id,
            "rotation_matrix": rotation_matrix,
            "translation_vector": translation_vector
        }
    
    return camera_poses

def save_camera_poses(camera_poses, output_file="camera_poses.txt"):
    """
    Save extracted camera poses to a text file.
    
    -input:
        camera_poses: dict
            Dictionary mapping image names to their corresponding poses.
            Each pose contains:
                - image_id: int
                - rotation_matrix: ndarray (3x3)
                - translation_vector: ndarray (3,).
        output_file: str, optional (default="camera_poses.txt")
            Path to the output file where camera poses will be saved.

    -output:
        None
        Writes the camera poses to the specified output file in a readable format.
    """
    with open(output_file, 'w') as file:
        for image_name, pose in camera_poses.items():
            file.write(f"Image: {image_name}\n")
            file.write(f"Image ID: {pose['image_id']}\n")
            file.write("Rotation Matrix:\n")
            file.write(f"{pose['rotation_matrix']}\n")
            file.write("Translation Vector:\n")
            file.write(f"{pose['translation_vector']}\n")
            file.write("\n")



#######################
# 3D points functions #
#######################
def load_points3D(file_path):
    """
    Load 3D points from COLMAP's points3D.txt file.
    
    Parameters:
        file_path (str): Path to points3D.txt.
    
    Returns:
        points (np.ndarray): Array of 3D points of shape (N, 3), where N is the number of points.
    """
    points = []
    color = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('#') or not line.strip():  # Skip comments and empty lines
                continue
            elements = line.split()
            x, y, z = map(float, elements[1:4])  # Extract X, Y, Z
            r, g, b = map(int, elements[4:7])
            points.append([x, y, z])
            color.append([r, g, b])
    return np.array(points), np.array(color)


