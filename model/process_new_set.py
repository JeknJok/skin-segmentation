import os
import cv2
import numpy as np
from tqdm import tqdm

# === CONFIGURATION ===
ROOT_DIR = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\Fasseg-DB-v2019"
OUTPUT_IMAGE_DIR = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\new_dataset\images"
OUTPUT_MASK_DIR = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\new_dataset\masks"

# Ensure output directories exist
os.makedirs(OUTPUT_IMAGE_DIR, exist_ok=True)
os.makedirs(OUTPUT_MASK_DIR, exist_ok=True)


def find_all_bmp_files(root_dir):
    """
    Recursively scans **ALL** directories and finds BMP images/masks.
    Collects images from any `*RGB*` folder and masks from any `*Labels*` folder.

    Returns:
        images (dict): Mapping of base filenames to image file paths.
        masks (dict): Mapping of base filenames to mask file paths.
    """
    images = {}
    masks = {}

    print("\n🔍 Scanning ALL directories...\n")
    for dirpath, _, filenames in os.walk(root_dir):
        print(f"📂 Checking: {dirpath}")

        # Process files within the folder
        for file in filenames:
            if file.lower().endswith((".bmp", ".png", ".jpg", ".jpeg")):  # Ensure all formats are caught
                file_path = os.path.join(dirpath, file)
                base_name = os.path.splitext(file)[0]  # Extract filename without extension

                # Normalize case for folder detection
                lower_path = dirpath.lower()

                # Store images from any folder containing "RGB"
                if "rgb" in lower_path:
                    if base_name in images:
                        print(f"⚠️ Duplicate image detected: {file_path}")
                    images[base_name] = file_path

                # Store masks from any folder containing "Labels"
                elif "labels" in lower_path:
                    if base_name in masks:
                        print(f"⚠️ Duplicate mask detected: {file_path}")
                    masks[base_name] = file_path

    print(f"\n📸 Total Images Found: {len(images)}")
    print(f"🎭 Total Masks Found: {len(masks)}\n")

    return images, masks


def move_and_process_bmp_files(image_files, mask_files):
    """
    Moves and processes images/masks with **unique filenames** to avoid overwriting.
    Ensures each image has a **correctly paired mask**.
    """
    paired_files = set(image_files.keys()).intersection(set(mask_files.keys()))

    if not paired_files:
        print("❌ ERROR: No matching images and masks found! Possible mismatches.")
        unmatched_images = set(image_files.keys()) - set(mask_files.keys())
        unmatched_masks = set(mask_files.keys()) - set(image_files.keys())

        print(f"🔴 Unmatched Images: {len(unmatched_images)} → {unmatched_images}")
        print(f"🔴 Unmatched Masks: {len(unmatched_masks)} → {unmatched_masks}")

        return

    print(f"\n✅ Processing {len(paired_files)} matched image-mask pairs...\n")

    for index, base_name in enumerate(tqdm(paired_files, desc="Processing")):
        unique_name = f"Train_{index:05d}"  # Ensures unique filename for each pair

        img_path = image_files[base_name]
        mask_path = mask_files[base_name]

        print(f"🔗 Pairing: {img_path} ↔ {mask_path}")  # Debugging

        img_output_path = os.path.join(OUTPUT_IMAGE_DIR, unique_name + ".jpg")
        mask_output_path = os.path.join(OUTPUT_MASK_DIR, unique_name + ".png")

        process_image(img_path, img_output_path)
        process_ground_truth_mask(mask_path, mask_output_path)


def process_image(image_path, output_path):
    """ Converts image to JPG. """
    img = cv2.imread(image_path)
    if img is not None:
        cv2.imwrite(output_path, img, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        print(f"❌ ERROR: Could not read image {image_path}")


def process_ground_truth_mask(mask_path, output_path):
    """
    Converts masks:
    ✅ Detects **skin-colored** regions
    ✅ **Removes dark blue areas (eyes)**
    ✅ Saves output in grayscale
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
    if mask is None:
        print(f"❌ ERROR: Could not read mask {mask_path}")
        return

    hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)

    # Define skin color ranges
    yellow_lower = np.array([20, 100, 100])
    yellow_upper = np.array([40, 255, 255])

    blue_lower = np.array([90, 50, 50])
    blue_upper = np.array([130, 255, 255])

    dark_blue_lower = np.array([100, 150, 50])  # Dark blue areas (eyes)
    dark_blue_upper = np.array([140, 255, 255])

    # Detect skin (yellow + blue)
    yellow_mask = cv2.inRange(hsv_mask, yellow_lower, yellow_upper)
    blue_mask = cv2.inRange(hsv_mask, blue_lower, blue_upper)
    skin_mask = cv2.bitwise_or(yellow_mask, blue_mask)

    # Remove **dark blue (eyes)**
    dark_blue_mask = cv2.inRange(hsv_mask, dark_blue_lower, dark_blue_upper)
    skin_mask = cv2.bitwise_and(skin_mask, cv2.bitwise_not(dark_blue_mask))

    # Convert to binary
    skin_mask = np.where(skin_mask > 0, 255, 0).astype(np.uint8)

    cv2.imwrite(output_path, skin_mask)


# === EXECUTION ===
print("\n🚀 Starting dataset processing...\n")

image_files, mask_files = find_all_bmp_files(ROOT_DIR)

# 🛠 DEBUG: Print missing file pairs
if len(image_files) < 200:
    print(f"⚠️ WARNING: Only found {len(image_files)} images. Expected at least 200!")
if len(mask_files) < 200:
    print(f"⚠️ WARNING: Only found {len(mask_files)} masks. Expected at least 200!")

move_and_process_bmp_files(image_files, mask_files)

print("\n✅ Processing complete! All images & masks saved in:")
print(f"   📂 {OUTPUT_IMAGE_DIR}")
print(f"   📂 {OUTPUT_MASK_DIR}")
