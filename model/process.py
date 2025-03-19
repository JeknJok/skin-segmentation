# Re-import necessary modules after code execution state reset
import os
import tarfile
import scipy.io
import numpy as np
from PIL import Image
import shutil
import pandas as pd
import ace_tools as tools

# Redefine paths
archive_path = "/mnt/data/Sitting.tar.gz"
extracted_path = "/mnt/data/Sitting_extracted"
converted_path = "/mnt/data/Sitting_png_masks"

# Step 1: Extract the tar.gz file
with tarfile.open(archive_path, "r:gz") as tar:
    tar.extractall(path=extracted_path)

# Step 2: Locate the 'masks' folder inside the extracted content
masks_folder = None
for root, dirs, files in os.walk(extracted_path):
    if os.path.basename(root) == "masks":
        masks_folder = root
        break

# Prepare output directory
os.makedirs(converted_path, exist_ok=True)

# Step 3: Convert each .mat file to .png
converted_files = []
if masks_folder:
    for filename in os.listdir(masks_folder):
        if filename.endswith(".mat"):
            mat_path = os.path.join(masks_folder, filename)
            mat_contents = scipy.io.loadmat(mat_path)

            # Attempt to find the mask data (assuming it's the largest 2D array)
            mask_data = None
            for key, value in mat_contents.items():
                if isinstance(value, np.ndarray) and value.ndim == 2:
                    mask_data = value
                    break

            if mask_data is not None:
                # Normalize mask to 0-255 for PNG
                mask_normalized = (mask_data - mask_data.min()) / (mask_data.ptp() or 1) * 255
                mask_image = Image.fromarray(mask_normalized.astype(np.uint8))

                png_filename = os.path.splitext(filename)[0] + ".png"
                png_path = os.path.join(converted_path, png_filename)
                mask_image.save(png_path)
                converted_files.append(png_path)

# Step 4: Zip converted PNG masks for download
zip_path = "/mnt/data/Sitting_png_masks.zip"
shutil.make_archive(zip_path.replace('.zip', ''), 'zip', converted_path)

# Display result to user
tools.display_dataframe_to_user("Converted PNG Masks", 
    pd.DataFrame({"PNG Mask Files": [os.path.basename(p) for p in converted_files]}))

zip_path
