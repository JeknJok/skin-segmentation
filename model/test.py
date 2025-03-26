from unet_model import model, mean_iou, combined_loss
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np
import os

model = tf.keras.models.load_model(
    "model_skin_segmentation.keras",
    custom_objects={"combined_loss": combined_loss,"mean_iou": mean_iou}
)

def visualize_predictions(model, test_image_path, test_mask_path):
    """
    Visualizes image, ground truth mask, and predicted mask.
    """

    #check the file if exists
    if not os.path.exists(test_image_path):
        print(f"Error: Test image not found at {test_image_path}")
        return
    if not os.path.exists(test_mask_path):
        print(f"Error: Test mask not found at {test_mask_path}")
        return

    #load img
    image = cv2.imread(test_image_path)
    if image is None:
        print(f"Error: Could not read image {test_image_path}")
        return
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    #load mask
    true_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    if true_mask is None:
        print(f"Error: Could not read mask {test_mask_path}")
        return

    true_mask = true_mask / 255.0
    true_mask = np.where(true_mask > 0.5, 1.0, 0.0)

    #preprocess img
    input_image = cv2.resize(image, (256, 256)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    #run model
    raw_pred_mask = model.predict(input_image)[0, :, :, 0]
    binary_mask = (raw_pred_mask > np.mean(raw_pred_mask)).astype(np.uint8) * 255

    cv2.imwrite("predicted_mask.png", binary_mask)

    print("Raw Prediction Min:", np.min(raw_pred_mask))
    print("Raw Prediction Max:", np.max(raw_pred_mask))
    print("Raw Prediction Mean:", np.mean(raw_pred_mask))
    print(f"Predicted Mask Min: {binary_mask.min()}, Max: {binary_mask.max()}, Mean: {binary_mask.mean()}")

    # Plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1), plt.imshow(image_rgb), plt.title("Original Image")
    plt.subplot(1, 4, 2), plt.imshow(true_mask, cmap="gray"), plt.title("True Mask")
    plt.subplot(1, 4, 3), plt.imshow(binary_mask, cmap="gray"), plt.title("Resized Predicted Mask")
    plt.subplot(1, 4, 4), plt.imshow(raw_pred_mask, cmap="jet"), plt.title("Raw Predicted Mask")
    plt.show()

visualize_predictions(model,
                      r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\dataset_small_500\fullbody\img\00a4ba6f18cc6c5d618b5f0993785210.jpg",
                      r"C:\Users\neall\Documents\2. University Stuff\2.SFU files\1Spring 2025\CMPT 340 - Biomedical Computing\project\model\dataset_small_500\fullbody\img\00a4ba6f18cc6c5d618b5f0993785210.jpg")