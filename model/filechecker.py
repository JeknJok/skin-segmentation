import os
import tensorflow as tf

print(tf.config.list_physical_devices('GPU')) 

#path
img_dir = r"Face_Dataset\images\face_photo"
mask_dir = r"Face_Dataset\masks\masks_face_photo"

# disp all
image_files = os.listdir(img_dir)
print(f"Total images: {len(image_files)}")
print("images:", image_files[:32])

# specific
missing_file = "0520962400.jpg"
if missing_file in image_files:
    print(" The file exists.")
else:
    print("404")