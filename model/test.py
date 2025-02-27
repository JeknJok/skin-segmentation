from unet_model import model,iou_loss
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
import numpy as np

model = tf.keras.models.load_model("saved_models/model_skin_segmentation.keras")

def visualize_predictions(model, test_image_path, test_mask_path):
    """
    testing program
    how can i make this an actual test?
    maybe only God knows
    """
    
    image = cv2.imread(test_image_path)
    mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    input_image = cv2.resize(image, (256, 256)) / 255.0
    input_image = np.expand_dims(input_image, axis=0)

    true_mask = cv2.imread(test_mask_path, cv2.IMREAD_GRAYSCALE)
    true_mask = true_mask / 255.0
    true_mask = np.where(true_mask > 0.5, 1.0, 0.0) 

    raw_pred_mask = model.predict(input_image)[0, :, :, 0]
    binary_mask = (raw_pred_mask > 0.02).astype(np.uint8) * 255  

    cv2.imwrite("predicted_mask.png", binary_mask)

    print("Raw Prediction Min:", np.min(raw_pred_mask))
    print("Raw Prediction Max:", np.max(raw_pred_mask))
    print("Raw Prediction Mean:", np.mean(raw_pred_mask))
    print(f"Predicted Mask Min: {binary_mask.min()}, Max: {binary_mask.max()}, Mean: {binary_mask.mean()}")
    
    #plot the results
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 4, 1), plt.imshow(image_rgb), plt.title("Original Image")
    plt.subplot(1, 4, 2), plt.imshow(true_mask, cmap="gray"), plt.title("True Mask")
    plt.subplot(1, 4, 3), plt.imshow(binary_mask, cmap="gray"), plt.title("Resized Predicted Mask")
    plt.subplot(1, 4, 4), plt.imshow(raw_pred_mask, cmap="jet"), plt.title("Raw Predicted Mask")
    plt.show()
    
visualize_predictions(model, 
                      r"Face_Dataset\images\face_photo\josh-hartnett-Poster-thumb.jpg",
                      r"Face_Dataset\masks\masks_face_photo\josh-hartnett-Poster-thumb.png")
    
    