from dataset import SkinDataset
from unet_model import unet_model, tversky_loss
import tensorflow as tf
from tensorflow.keras.optimizers import Adam

# Dataset paths
#face (single face)
img_dir = r"Face_Dataset\images\face_photo"
mask_dir = r"Face_Dataset\masks\masks_face_photo"

#family photo (many people)
img_dir2 = r"Face_Dataset\images\family_photo"
mask_dir2 = r"Face_Dataset\masks\masks_family_photo"

# Load dataset and convert to TensorFlow Dataset
dataset = SkinDataset(img_dir, mask_dir)
#train_dataset, val_dataset = dataset.get_train_val_datasets()
train_dataset =  dataset.get_train_val_datasets()

# init model
model = unet_model(input_shape=(256, 256, 3), num_classes=1)

# compile model
#use learning rate scheduler to improve quality
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001, decay_steps=5000, decay_rate=0.95
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

# Compile model with Tversky loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss=tversky_loss,
              metrics=["accuracy"])

#with tf.device('/GPU 0'): 
history = model.fit(
    train_dataset.batch(16), 
    epochs=60,
    validation_data=None
)

#save the model
model.save("saved_models/unet_skin_segmentation.keras")
print("Model done!")