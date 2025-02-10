from dataset import SkinDataset
from unet_model import unet_model
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
train_dataset, val_dataset = dataset.get_train_val_datasets()

# init model
model = unet_model(input_shape=(256, 256, 3), num_classes=1)

# compile model
#use learning rate scheduler to improve quality
lr = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=10000,
    decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr)
model.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"])

#train
#with tf.device('/GPU 0'): 
history = model.fit(train_dataset, 
                        epochs=1, 
                        validation_data=None,
                        callbacks=[tf.keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

#save the model
model.save("saved_models/unet_skin_segmentation.keras")
print("Model done!")