import json
import sqlite3
import os
import numpy as np


def initialize_database(db_name="database.db"):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Cameras table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Cameras (
        camera_id INTEGER PRIMARY KEY AUTOINCREMENT,
        model TEXT NOT NULL,
        width INTEGER NOT NULL,
        height INTEGER NOT NULL,
        params TEXT NOT NULL,
        prior_focal_length REAL DEFAULT 0
    );
    """)

    # Images table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Images (
        image_id INTEGER PRIMARY KEY AUTOINCREMENT,
        name TEXT NOT NULL UNIQUE,
        camera_id INTEGER NOT NULL,
        FOREIGN KEY (camera_id) REFERENCES Cameras(camera_id)
    );
    """)

    # Keypoints table with foreign key
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Keypoints (
        ImageID INTEGER,
        KeypointID INTEGER,
        row REAL,
        col REAL,
        PRIMARY KEY (ImageID, KeypointID),
        FOREIGN KEY (ImageID) REFERENCES Images(image_id)
    );
    """)


     # Matches table with pair_id
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Matches (
        pair_id TEXT NOT NULL,
        ImageID1 INTEGER,
        KeypointID1 INTEGER,
        ImageID2 INTEGER,
        KeypointID2 INTEGER,
        PRIMARY KEY (pair_id, ImageID1, KeypointID1, ImageID2, KeypointID2),
        FOREIGN KEY (pair_id) REFERENCES two_view_geometries(pair_id)
    );
    """)


    # 3D Points table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Points3D (
        Point3DID INTEGER PRIMARY KEY,
        X REAL,
        Y REAL,
        Z REAL,
        R INTEGER,
        G INTEGER,
        B INTEGER,
        Error REAL
    );
    """)

    # Tracks table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS Tracks (
        Point3DID INTEGER,
        ImageID INTEGER,
        KeypointID INTEGER,
        PRIMARY KEY (Point3DID, ImageID, KeypointID),
        FOREIGN KEY (Point3DID) REFERENCES Points3D(Point3DID)
    );
    """)


    # Two View Geometries table
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS two_view_geometries (
        pair_id TEXT PRIMARY KEY,
        rows INTEGER,  -- Number of matches
        F TEXT,        -- Fundamental matrix in JSON format
        E TEXT,        -- Essential matrix in JSON format (optional)
        H TEXT,        -- Homography matrix in JSON format (optional)
        qvec TEXT,     -- Rotation (quaternion) in JSON format (optional)
        tvec TEXT      -- Translation vector in JSON format (optional)
    );
    """)


    conn.commit()
    conn.close()
    print(f"Database initialized at {db_name}")


def add_camera(db_name, model, width, height, params, prior_focal_length):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Insert camera and return its ID
    cursor.execute("""
    INSERT INTO Cameras (model, width, height, params, prior_focal_length)
    VALUES (?, ?, ?, ?, ?);
    """, (model, width, height, params, prior_focal_length))

    camera_id = cursor.lastrowid
    conn.commit()
    conn.close()
    print(f"Camera added with ID: {camera_id}")
    return camera_id


def add_image(db_name, image_name, camera_id):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Insert or retrieve image ID
    cursor.execute("""
    INSERT OR IGNORE INTO Images (name, camera_id)
    VALUES (?, ?);
    """, (image_name, camera_id))

    # Retrieve the image ID
    cursor.execute("SELECT image_id FROM Images WHERE name = ?;", (image_name,))
    image_id = cursor.fetchone()[0]

    conn.commit()
    conn.close()
    print(f"Image added with ID: {image_id}")
    return image_id


def load_keypoints(db_name, image_name, keypoints_file):
    import sqlite3
    import numpy as np

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Retrieve the image ID
    cursor.execute("SELECT image_id FROM Images WHERE name = ?;", (image_name,))
    image_id_row = cursor.fetchone()
    if image_id_row is None:
        print(f"Image {image_name} not found in the database.")
        conn.close()
        return

    image_id = image_id_row[0]

    # Load keypoints from the npz file
    if keypoints_file.endswith(".npy"):  # If directly a .npy file
        keypoints_data = np.load(keypoints_file)
    else:
        keypoints_data = np.load(keypoints_file)["keypoints"]

    # Ensure keypoints are float (convert if necessary)
    keypoints_data = np.asarray(keypoints_data, dtype=np.float32)

    # Insert keypoints into the database
    for keypoint_id, (row, col) in enumerate(keypoints_data):
        cursor.execute("""
        INSERT OR IGNORE INTO Keypoints (ImageID, KeypointID, row, col)
        VALUES (?, ?, ?, ?);
        """, (image_id, keypoint_id, float(row), float(col)))

    conn.commit()
    conn.close()
    print(f"Keypoints loaded for {image_name}: {len(keypoints_data)} keypoints.")


def load_matches(db_name, matches_file, image_name1, image_name2):
    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    # Retrieve image IDs
    cursor.execute("SELECT image_id FROM Images WHERE name = ?;", (image_name1,))
    image_id1_row = cursor.fetchone()
    if image_id1_row is None:
        print(f"Image {image_name1} not found in the database.")
        conn.close()
        return
    image_id1 = image_id1_row[0]

    cursor.execute("SELECT image_id FROM Images WHERE name = ?;", (image_name2,))
    image_id2_row = cursor.fetchone()
    if image_id2_row is None:
        print(f"Image {image_name2} not found in the database.")
        conn.close()
        return
    image_id2 = image_id2_row[0]

    # Load matches
    matches = np.load(matches_file)

    # Ensure matches are in the correct format
    if "matches" in matches:
        matches_data = matches["matches"]
    elif matches_file.endswith(".npy"):
        matches_data = matches
    else:
        print(f"Invalid matches file format: {matches_file}")
        conn.close()
        return

    # Ensure matches data is an array of pairs
    matches_data = np.asarray(matches_data, dtype=np.int32)

    # Generate pair_id
    pair_id = f"{image_name1}_{image_name2}"

    # Insert matches into the database
    for kp0, kp1 in matches_data:
        kp0 = int(kp0)
        kp1 = int(kp1)
        cursor.execute("""
        INSERT OR IGNORE INTO Matches (pair_id, ImageID1, KeypointID1, ImageID2, KeypointID2)
        VALUES (?, ?, ?, ?, ?);
        """, (pair_id, image_id1, kp0, image_id2, kp1))

    # Add F to two_view_geometries
    F_matrix = np.load(matches_file)["F"]
    F_json = json.dumps(F_matrix.tolist())
    num_matches = len(matches_data)

    cursor.execute("""
    INSERT OR REPLACE INTO two_view_geometries (
        pair_id, rows, F
    ) VALUES (?, ?, ?);
    """, (pair_id, num_matches, F_json))

    conn.commit()
    conn.close()
    print(f"Matches loaded for {image_name1} and {image_name2}, {len(matches_data)} matches.")
    print(f"Two-view geometry added for pair: {pair_id}")

def add_two_view_geometry(db_name, image_name1, image_name2, F_matrix, num_matches):
    """
    Insert F matrix and other optional data into the two_view_geometries table.
    :param db_name: Path to the database.
    :param image_name1: First image name.
    :param image_name2: Second image name.
    :param F_matrix: Fundamental matrix as a numpy array.
    :param num_matches: Number of matches between the image pair.
    """
    pair_id = f"{image_name1}_{image_name2}"
    F_json = json.dumps(F_matrix.tolist())

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
    INSERT OR REPLACE INTO two_view_geometries (
        pair_id, rows, F
    ) VALUES (?, ?, ?);
    """, (pair_id, num_matches, F_json))

    conn.commit()
    conn.close()
    print(f"Two-view geometry added for pair: {pair_id}")


def get_two_view_geometry(db_name, image_name1, image_name2):
    """
    Retrieve F matrix from the database for a given pair of images.
    :param db_name: Path to the database.
    :param image_name1: First image name.
    :param image_name2: Second image name.
    :return: F matrix as a numpy array or None.
    """
    pair_id = f"{image_name1}_{image_name2}"

    conn = sqlite3.connect(db_name)
    cursor = conn.cursor()

    cursor.execute("""
    SELECT F FROM two_view_geometries WHERE pair_id = ?;
    """, (pair_id,))
    result = cursor.fetchone()

    conn.close()

    if result:
        F_json = result[0]
        return np.array(json.loads(F_json))
    else:
        print(f"No two-view geometry found for pair: {pair_id}")
        return None


if __name__ == "__main__":
    # Define project structure
    project_root_dir = "/Users/ignaciopastorebenaim/Documents/MGRCV/TPs/CV/Colon_assignment/own_projects"
    seq_name = "Seq_035"
    type = "toy"
    db_name = "database.db"
    project_dir = os.path.join(project_root_dir, seq_name, type)
    database_path = os.path.join(project_dir,db_name)

    # Initialize database
    initialize_database(database_path)

    # Add camera
    camera_id = add_camera(
        database_path,
        model="PINHOLE",
        width=960,
        height=720,
        params="727.1851, 728.5954, 668.1817, 507.4003",
        prior_focal_length=0
    )

    # Load images in alphabetical order
    images_dir = os.path.join(project_dir, "images")
    image_files = sorted(os.listdir(images_dir), key=lambda x: (x[:7], x[7:]))
    for image_file in image_files:
        image_name = os.path.splitext(image_file)[0]
        add_image(database_path, image_name, camera_id)

    # Load keypoints
    # First index=0 as in COLMAP
    keypoints_dir = os.path.join(project_dir, "keypoints")
    for keypoints_file in os.listdir(keypoints_dir):
        image_name = os.path.splitext(keypoints_file)[0]
        keypoints_path = os.path.join(keypoints_dir, keypoints_file)
        load_keypoints(database_path, image_name, keypoints_path)

    # Load matches
    matches_dir = os.path.join(project_dir, "matches")
    for matches_file in os.listdir(matches_dir):
       
        parts = matches_file.split("_")
        image_name1 = "_".join(parts[:2])
        image_name2 = "_".join(parts[2:4])
        matches_path = os.path.join(matches_dir, matches_file)
        load_matches(database_path, matches_path, image_name1, image_name2)
        



