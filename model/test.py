from unet_model import unet_model
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf

# Load the trained model
model = tf.keras.models.load_model("saved_models/unet_skin_segmentation.keras")

def visualize_predictions(model, test_image_path, test_mask_path):
    """
    Visualize predictions on a test image.
    """
    # Load test image and mask
    image = cv2.imread(test_image_path)
    mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

    # Preprocess image
    input_image = cv2.resize(image, (256, 256))
    input_image = input_image / 255.0
    input_image = input_image[tf.newaxis, ...]

    # Predict
    pred_mask = model.predict(input_image)[0]
    pred_mask = (pred_mask > 0.5).astype("uint8") * 255

    # Plot results
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 3, 1), plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)), plt.title("Original Image")
    plt.subplot(1, 3, 2), plt.imshow(mask, cmap="gray"), plt.title("True Mask")
    plt.subplot(1, 3, 3), plt.imshow(pred_mask, cmap="gray"), plt.title("Predicted Mask")
    plt.show()

# Test the model
visualize_predictions(model, r"Face_Dataset\images\face_photo\0520962400.jpg", r"Face_Dataset\masks\masks_face_photo\0520962400.png")