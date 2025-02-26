from unet_model import model, iou_loss, iou_loss, mean_iou
import tensorflow as tf
from dataset import SkinDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from plot import PlotCallback
tf.config.run_functions_eagerly(True)

#datasets 
img_dir = r"Face_Dataset\images\face_photo"
mask_dir = r"Face_Dataset\masks\masks_face_photo"

dataset = SkinDataset(img_dir, mask_dir,val_split=0.2)
train_dataset, val_dataset = dataset.get_train_val_datasets()
steps_per_epoch = len(dataset.train_img_paths) // 8 # match batch size

#check sample image stats
sample_img, sample_mask = next(iter(train_dataset))
print("Sample Image Shape:", sample_img.shape)#expected (batch_size, 256, 256, 3)
print("Sample Mask Shape:", sample_mask.shape)
print("Sample Image Stats:")
print("Min:", np.min(sample_img.numpy()), "Max:", np.max(sample_img.numpy()), "Mean:", np.mean(sample_img.numpy()))

print("\nSample Mask Stats:")
print("Min:", np.min(sample_mask.numpy()), "Max:", np.max(sample_mask.numpy()), "Unique:", np.unique(sample_mask.numpy()))
#=========================================================================================================================
#remove old model
model_path = "saved_models/model_skin_segmentation.keras"
if os.path.exists(model_path):
    os.remove(model_path)
    print("Old model removed.")
else:
    print("No existing model found. Proceeding with training.")

#initialize model
model = model(input_shape=(256, 256, 3), num_classes=1)
#model.summary()
#for layer in model.layers:
    #print(layer.name, "Trainable:", layer.trainable)
# expected
#conv1_relu Trainable: False
#conv2_block3_out Trainable: False
#conv3_block4_out Trainable: False
#conv4_block6_out Trainable: True
#conv5_block3_out Trainable: True
#=========================================================================================================================
# TRAIN MODEL IN 2 PHASES
# I choose to train the model in two phases to give more effect towards my finetuning of the model.
# the phases are: (1) training decoder only, and then  (2) the encoder.



# TRAINING DECODER ONLY
# encoder has pre-trained weights from ImageNet, so we freeze it and train
# only the decoder here, since the decoder needs to learn from scratch.
print("NOW TRAINING DECODER")
for layer in model.layers:
    if "conv1" in layer.name or "conv2" in layer.name or "conv3" in layer.name:
        layer.trainable = False
    else:
        layer.trainable = True 

# OPTIMIZER with exponential learning rate decay
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.96,
    staircase=True,
)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer, loss=iou_loss, metrics=[mean_iou])

history = model.fit(train_dataset, 
                    epochs=150,
                    validation_data=None
                    )

# UNFREEZE ENCODER AND TRAIN FULLY
# Now that the decoder is trained, we "unfreeze" deeper layers 
# in resnet50 to fine-tune the feature.
print("NOW TRAINING ENCODER")
for layer in model.layers:
    layer.trainable = True
optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4) 
model.compile(optimizer, loss=iou_loss, metrics=[mean_iou])
history_finetune = model.fit(train_dataset, 
                             epochs=50,
                             validation_data=None, 
                             callbacks=[PlotCallback()]
                             )

model.save("saved_models/model_skin_segmentation.keras")

print("done")
