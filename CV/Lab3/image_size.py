import cv2

# Load the image using OpenCV
image_path = './data/image1.png' 
image = cv2.imread(image_path)

# Get the dimensions of the image
height, width, channels = image.shape

print(f"Image width: {width} pixels")
print(f"Image height: {height} pixels")

python ./match_pairs.py --resize 752 --superglue indoor --max_keypoints 1024 --input_dir ../data --input_pairs ../data/images_paired.txt --output_dir ./SuperGlue_output --viz
