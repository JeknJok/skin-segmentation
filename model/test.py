from unet_model import combined_loss
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("saved_models/unet_skin_segmentation.keras",
                                   custom_objects={"combined_loss": combined_loss})

def visualize_predictions(model, test_image_path, test_mask_path):
    image = cv2.imread(test_image_path)
    mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_image = cv2.resize(image, (256, 256)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    true_mask = cv2.resize(mask, (256, 256)) / 255.0

    raw_pred_mask = model.predict(input_image)[0, :, :, 0]
    print(f"Predicted Mask Min: {raw_pred_mask.min()}, Max: {raw_pred_mask.max()}")

    pred_mask = (raw_pred_mask > 0.09).astype(np.uint8) * 255

    pred_mask_resized = cv2.resize(pred_mask, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_NEAREST)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1), plt.imshow(image_rgb), plt.title("Original Image")
    plt.subplot(1, 4, 2), plt.imshow(true_mask, cmap="gray"), plt.title("True Mask")
    plt.subplot(1, 4, 3), plt.imshow(pred_mask_resized, cmap="gray"), plt.title("Resized Predicted Mask")
    plt.subplot(1, 4, 4), plt.imshow(raw_pred_mask, cmap="jet"), plt.title("Raw Predicted Mask")
    plt.show()

visualize_predictions(model, 
                      r"Face_Dataset\images\face_photo\josh-hartnett-Poster-thumb.jpg",
                      r"Face_Dataset\masks\masks_face_photo\josh-hartnett-Poster-thumb.png")
