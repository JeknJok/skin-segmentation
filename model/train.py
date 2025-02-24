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

#load dataset and convert to TensorFlow dataset
dataset = SkinDataset(img_dir, mask_dir)

train_dataset = dataset.get_train_val_datasets().take(20)
train_dataset = train_dataset.prefetch(tf.data.AUTOTUNE)
# init model
model = unet_model(input_shape=(256, 256, 3), num_classes=1)

# compile model
#use learning rate scheduler to improve quality
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.007, decay_steps=5000, decay_rate=0.9
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

early_stopping = tf.keras.callbacks.EarlyStopping(monitor="loss", patience=10, restore_best_weights=True)

# Compile model with Tversky loss
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
              loss="binary_crossentropy",
              metrics=["accuracy"])

for img, mask in train_dataset.take(1):
    print("Image batch shape:", img.shape)
    print("Mask batch shape:", mask.shape)

#with tf.device('/GPU 0'): 
history = model.fit(
    train_dataset, 
    epochs=120,
    validation_data=None
)

#save the model
model.save("saved_models/unet_skin_segmentation.keras")
print("Model done!")