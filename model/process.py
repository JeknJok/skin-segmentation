import os
import io
from rembg import remove
from PIL import Image
import cv2
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from unet_model import model, mean_iou, combined_loss

# I USED LLM (CHATGPT) TO GENERATE THIS CODE - MERGING test.py and remove_bg.py

# Load your trained model
model = tf.keras.models.load_model(
    "model_skin_segmentation.keras",
    custom_objects={"combined_loss": combined_loss, "mean_iou": mean_iou}
)

def remove_background(input_path):
    with open(input_path, 'rb') as i:
        input_data = i.read()
    result = remove(input_data)
    output_image = Image.open(io.BytesIO(result)).convert("RGB")
    return output_image

def visualize_predictions(model, image_pil, test_mask_path, save_prefix="output"):
    """
    Accepts a PIL image (background removed), runs prediction, visualizes, and saves cutout.
    """
    # Convert to OpenCV image
    image = cv2.cvtColor(np.array(image_pil), cv2.COLOR_RGB2BGR)
    image_rgb = np.array(image_pil)
    original_height, original_width = image.shape[:2]

    # Load true mask
    if not os.path.exists(test_mask_path):
        print(f"Error: Test mask not found at {test_mask_path}")
        return
    true_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    if true_mask is None:
        print(f"Error: Could not read mask {test_mask_path}")
        return
    true_mask = true_mask / 255.0
    true_mask = np.where(true_mask > 0.5, 1.0, 0.0)

    # Preprocess input for model
    resized_input = cv2.resize(image, (256, 256)) / 255.0
    resized_input = np.expand_dims(resized_input, axis=0)

    # Predict
    raw_pred_mask = model.predict(resized_input)[0, :, :, 0]
    resized_pred = cv2.resize(raw_pred_mask, (original_width, original_height))
    binary_mask = (resized_pred > np.mean(resized_pred)).astype(np.uint8) * 255

    # Save predicted mask
    cv2.imwrite(f"{save_prefix}_predicted_mask.png", binary_mask)

    # Create cutout
    mask_3channel = cv2.merge([binary_mask, binary_mask, binary_mask])
    cutout = cv2.bitwise_and(image, mask_3channel)

    # Save cutout
    cv2.imwrite(f"{save_prefix}_cutout.png", cutout)

    print("Raw Prediction Min:", np.min(resized_pred))
    print("Raw Prediction Max:", np.max(resized_pred))
    print("Raw Prediction Mean:", np.mean(resized_pred))
    print(f"Predicted Mask Min: {binary_mask.min()}, Max: {binary_mask.max()}, Mean: {binary_mask.mean()}")

    # Plot
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 5, 1), plt.imshow(image_rgb), plt.title("BG Removed Image")
    plt.subplot(1, 5, 2), plt.imshow(true_mask, cmap="gray"), plt.title("True Mask")
    plt.subplot(1, 5, 3), plt.imshow(binary_mask, cmap="gray"), plt.title("Predicted Mask")
    plt.subplot(1, 5, 4), plt.imshow(resized_pred, cmap="jet"), plt.title("Raw Prediction")
    plt.subplot(1, 5, 5), plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)), plt.title("Cutout Result")
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    test_image_path = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\test-img\360_F_255139401_6yGRr1YEq1ybRTOqodsr3H8WDX25HVjE.jpg"         # Input image (with background)
    test_mask_path = r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\test-img\360_F_255139401_6yGRr1YEq1ybRTOqodsr3H8WDX25HVjE.jpg"   # Ground truth mask

    # Step 1: Remove background
    bg_removed_pil = remove_background(test_image_path)

    # Step 2: Segment skin and visualize
    visualize_predictions(model, bg_removed_pil, test_mask_path, save_prefix="sample")
