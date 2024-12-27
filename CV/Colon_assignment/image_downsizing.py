import cv2
import os
import matplotlib.pyplot as plt


seq_name = "Seq_035"
seq_path = f"./data/{seq_name}"
output_dir = f"./data/{seq_name}/resized/"

os.makedirs(output_dir, exist_ok=True)

for image in os.listdir(seq_path):
    if not image.endswith(".png"):
        continue
    image_path = os.path.join(seq_path, image)
    image_cv = cv2.imread(image_path)
    image_resized = cv2.resize(image_cv, (960, 720))
    output_path = os.path.join(output_dir, image)
    cv2.imwrite(output_path, image_resized)
    print(f"Resized {output_path}")









