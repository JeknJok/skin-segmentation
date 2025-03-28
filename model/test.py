from unet_model import model, mean_iou, combined_loss
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import os

# Load model
model = tf.keras.models.load_model(
    "model_skin_segmentation.keras",
    custom_objects={"combined_loss": combined_loss, "mean_iou": mean_iou}
)

def visualize_predictions(model, test_image_path, test_mask_path):
    """
    Visualizes and saves prediction results and cutout using the skin mask.
    """

    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return
    if not os.path.exists(test_mask_path):
        print(f"Error: Test mask not found at {test_mask_path}")
        return

    # Load original image
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not read image {test_image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    original_height, original_width = image.shape[:2]

    # Load true mask
    true_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    if true_mask is None:
        print(f"Error: Could not read mask {test_mask_path}")
        return
    true_mask = true_mask / 255.0
    true_mask = np.where(true_mask > 0.5, 1.0, 0.0)

    # Preprocess input image
    resized_input = cv2.resize(image, (256, 256)) / 255.0
    resized_input = np.expand_dims(resized_input, axis=0)

    # Predict mask
    raw_pred_mask = model.predict(resized_input)[0, :, :, 0]
    resized_pred = cv2.resize(raw_pred_mask, (original_width, original_height))
    binary_mask = (resized_pred > np.mean(resized_pred)).astype(np.uint8) * 255

    # Save predicted mask
    cv2.imwrite("predicted_mask.png", binary_mask)

    # Create cutout: apply binary mask to original image
    mask_3channel = cv2.merge([binary_mask, binary_mask, binary_mask])
    cutout = cv2.bitwise_and(image, mask_3channel)

    # Save cutout result
    cv2.imwrite("cutout_result.png", cutout)

    print("Raw Prediction Min:", np.min(resized_pred))
    print("Raw Prediction Max:", np.max(resized_pred))
    print("Raw Prediction Mean:", np.mean(resized_pred))
    print(f"Predicted Mask Min: {binary_mask.min()}, Max: {binary_mask.max()}, Mean: {binary_mask.mean()}")

    # Plot results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 5, 1), plt.imshow(image_rgb), plt.title("Original Image")
    plt.subplot(1, 5, 2), plt.imshow(true_mask, cmap="gray"), plt.title("True Mask")
    plt.subplot(1, 5, 3), plt.imshow(binary_mask, cmap="gray"), plt.title("Predicted Mask")
    plt.subplot(1, 5, 4), plt.imshow(resized_pred, cmap="jet"), plt.title("Raw Prediction")
    plt.subplot(1, 5, 5), plt.imshow(cv2.cvtColor(cutout, cv2.COLOR_BGR2RGB)), plt.title("Cutout Result")
    plt.tight_layout()
    plt.show()

visualize_predictions(model,
                      r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\no bg\fullbody\img\00f8f5f0ed311b44eb6654091001b655.jpg",
                      r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\no bg\fullbody\img\00f8f5f0ed311b44eb6654091001b655.jpg")