from unet_model import model, mean_iou, combined_loss
import tensorflow as tf
from dataset import SkinDataset
import os
import numpy as np
import matplotlib.pyplot as plt
from plot import PlotCallback
from sklearn.utils import shuffle
tf.config.run_functions_eagerly(True)

# === UPSAMPLING CONFIG ===
dataset_root = "/kaggle/input/test-dataset-900"

#upsampling factor
# Upsampling literally duplicates imgs of categories that are less than others, matching the other's numbers to avoid bias.

category_upsampling = {
    "face": 1,
    "fullbody": 1,
}

all_img_paths = []
all_mask_paths = []

# === Combine categories ===
for category, factor in category_upsampling.items():
    img_dir = os.path.join(dataset_root, category, "img")
    mask_dir = os.path.join(dataset_root, category, "masks")

    if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
        print(f" Skipping '{category}' (directory missing)")
        continue

    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg"))])
    mask_files = sorted([f for f in os.listdir(mask_dir) if f.endswith(".png")])

    matched_files = sorted(set(f.rsplit(".", 1)[0] for f in img_files) & set(f.rsplit(".", 1)[0] for f in mask_files))

    img_paths = [os.path.join(img_dir, f + ".jpg") for f in matched_files]
    mask_paths = [os.path.join(mask_dir, f + ".png") for f in matched_files]

    print(f"Category: {category} - Found {len(matched_files)} pairs, Upsample factor: {factor}")

    # Apply upsampling
    all_img_paths.extend(img_paths * factor)
    all_mask_paths.extend(mask_paths * factor)

# Shuffle all paths to avoid bias
all_img_paths, all_mask_paths = shuffle(all_img_paths, all_mask_paths, random_state=42)

print(f"\nTotal combined images: {len(all_img_paths)}, masks: {len(all_mask_paths)}")

# === Save combined lists to temp directory ===
combined_img_dir = "/kaggle/working/combined_dataset/img"
combined_mask_dir = "/kaggle/working/combined_dataset/masks"
os.makedirs(combined_img_dir, exist_ok=True)
os.makedirs(combined_mask_dir, exist_ok=True)

# Symlink - no copying
for i, (img_path, mask_path) in enumerate(zip(all_img_paths, all_mask_paths)):
    img_dest = os.path.join(combined_img_dir, f"combined_{i:05d}.jpg")
    mask_dest = os.path.join(combined_mask_dir, f"combined_{i:05d}.png")
    os.symlink(img_path, img_dest)
    os.symlink(mask_path, mask_dest)

print("\n All combined image/mask symlinks created.")

##=========================================== END OF UPSAMPLING CONFIG ==========================================

batch_size = 8
dataset = SkinDataset(combined_img_dir, combined_mask_dir,img_size=(256, 256), batch_size=batch_size, val_split=0.2)
train_dataset, val_dataset = dataset.get_train_val_datasets()
steps_per_epoch = len(dataset.train_img_paths) // batch_size

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
model_path = "model_skin_segmentation.keras"
if os.path.exists(model_path):
    os.remove(model_path)
    print("Old model removed.")
else:
    print("No existing model found. Proceeding with training.")

#initialize model
model = model(input_shape=(256, 256, 3), num_classes=1)
#model.summary()
#=========================================================================================================================
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0005, clipvalue=1.0)
reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.5,
    patience=5,
    min_lr=1e-6,
    verbose=1
)
model.compile(optimizer=optimizer, loss=combined_loss, metrics=[mean_iou])

# start training - single phase
history = model.fit(train_dataset,
                    epochs=70,
                    validation_data=val_dataset,
                    callbacks=[PlotCallback("u2net_training.png"), reduce_lr]
                    )

model.save("model_skin_segmentation.keras", include_optimizer=False)
print("Training complete.")