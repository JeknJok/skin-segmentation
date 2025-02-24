from unet_model import model
import tensorflow as tf
from dataset import SkinDataset

#datasets 
img_dir = r"Face_Dataset\images\face_photo"
mask_dir = r"Face_Dataset\masks\masks_face_photo"

dataset = SkinDataset(img_dir, mask_dir)
train_dataset = dataset.get_train_val_datasets()

#initialize model
model = model(input_shape=(256, 256, 3), num_classes=1)
model.summary()

#train decoder block only
# encoder has pre-trained weights from ImageNet, so we freeze it and train
#only the decoder here, since the decoder needs to learn from scratch.
#
for layer in model.layers:
    if "conv5" in layer.name or "conv4" in layer.name:
        layer.trainable = True

optimizer = tf.keras.optimizers.Adam(learning_rate=0.001)

model.compile(optimizer=optimizer,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=["accuracy"])

history = model.fit(train_dataset, epochs=30, validation_data=None)

# Now that the decoder is trained, we "unfreeze" deeper layers 
# in resnet50 to fine-tune the feature.
#
for layer in model.layers:
    if "conv5" in layer.name or "conv4" in layer.name: 
        layer.trainable = True

optimizer_finetune = tf.keras.optimizers.Adam(learning_rate=0.0005) 

model.compile(optimizer=optimizer_finetune,
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
              metrics=["accuracy"])

history_finetune = model.fit(train_dataset, epochs=50, validation_data=None)

model.save("saved_models/model_skin_segmentation.keras")

print("done")
