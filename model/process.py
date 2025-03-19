import os
import cv2
import numpy as np

def process_skin_masks(base_dir, output_base_dir):
    """Iterates through all subdirectories, converting skin regions to white and non-skin regions to black, ensuring undergarments (light blue) and dark green regions are excluded while preserving all actual skin, including the head (light green)."""
    if not os.path.exists(output_base_dir):
        os.makedirs(output_base_dir)
    
    for root, _, files in os.walk(base_dir):
        relative_path = os.path.relpath(root, base_dir)
        output_dir = os.path.join(output_base_dir, relative_path)
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
        for mask_name in files:
            mask_path = os.path.join(root, mask_name)
            output_path = os.path.join(output_dir, mask_name)
            
            # Read the mask image
            mask = cv2.imread(mask_path, cv2.IMREAD_COLOR)
            if mask is None:
                print(f"Skipping invalid mask: {mask_path}")
                continue
            
            # Convert original mask to HSV color space for better color filtering
            hsv_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2HSV)
            
            # Define light blue color range (undergarments)
            lower_blue = np.array([90, 50, 50])
            upper_blue = np.array([130, 255, 255])
            
            # Define dark green color range
            lower_dark_green = np.array([35, 40, 40])
            upper_dark_green = np.array([85, 255, 255])
            
            # Define refined light green color range (head, avoiding overlaps)
            lower_light_green = np.array([50, 100, 100])
            upper_light_green = np.array([80, 255, 255])
            
            # Identify pixels in the undergarment and dark green ranges
            blue_mask = cv2.inRange(hsv_mask, lower_blue, upper_blue)
            dark_green_mask = cv2.inRange(hsv_mask, lower_dark_green, upper_dark_green)
            light_green_mask = cv2.inRange(hsv_mask, lower_light_green, upper_light_green)
            
            # Exclude undergarments and dark green areas
            exclusion_mask = cv2.bitwise_or(blue_mask, dark_green_mask)
            
            # Convert original mask to grayscale AFTER applying color-based filtering
            gray_mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
            gray_mask[exclusion_mask > 0] = 0  # Set excluded areas to black
            
            # Ensure light green (head) is included as white, avoiding incorrect inclusions
            head_mask = np.where(light_green_mask > 0, 255, 0).astype(np.uint8)
            gray_mask = cv2.bitwise_or(gray_mask, head_mask)
            
            # Convert remaining non-black pixels to white (skin regions)
            binary_mask = np.where(gray_mask > 30, 255, 0).astype(np.uint8)
            
            # Save the processed mask with the same name in the corresponding folder
            cv2.imwrite(output_path, binary_mask)
            print(f"Processed: {output_path}")

# Example usage
mask_base_directory = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\addition_dataset"
output_base_directory = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\process_masks"
process_skin_masks(mask_base_directory, output_base_directory)
