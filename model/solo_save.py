from dataset import SkinDataset
from unet_model import unet_model, combined_loss, dice_loss, dice_coef
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

model = unet_model(input_shape=(256, 256, 3), num_classes=1)
model.save("saved_models/unet_skin_segmentation.keras")