## dataset.py
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.model_selection import train_test_split
import tensorflow as tf

# funct for perspecitive warp
def apply_transform(image, warp_matrix, interpolation="BILINEAR"):
    transform_matrix = tf.reshape(warp_matrix, [8])  # Reshape to match TensorFlow format
    image_size = tf.shape(image)[0:2]

    return tf.raw_ops.ImageProjectiveTransformV2(
        images=tf.expand_dims(image, axis=0),
        transforms=tf.expand_dims(transform_matrix, axis=0),
        output_shape=image_size,
        interpolation=interpolation
    )[0]

def augment_data(image, mask):
    """
    Perform data augmentation on the image and mask together.
    The mask and image are first combined into one, and are given random modifications.
    Then they are split back to two and are returned.

    Input: mask, image
    Output, mask, image, the same ones at the same time.

    """
    # Random flip
    image = tf.image.random_flip_left_right(image)
    mask = tf.image.random_flip_left_right(mask)

    # Random brightness & contrast
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)

    # Random saturation & hue
    image = tf.image.random_saturation(image, lower=0.7, upper=1.3)
    image = tf.image.random_hue(image, max_delta=0.05)

    # Random scale
    scale = tf.random.uniform([], 0.9, 1.1)
    new_size = tf.cast(tf.convert_to_tensor([256.0, 256.0]) * scale, tf.int32)
    image = tf.image.resize(image, new_size)
    mask = tf.image.resize(mask, new_size)

    # Perspective warp
    warp_matrix = tf.random.uniform([8], minval=-0.1, maxval=0.1)
    image = apply_transform(image, warp_matrix, interpolation="BILINEAR")
    mask = apply_transform(mask, warp_matrix, interpolation="NEAREST")

    # Random rotation
    k = tf.random.uniform([], 0, 4, dtype=tf.int32)
    image = tf.image.rot90(image, k)
    mask = tf.image.rot90(mask, k)

    # Add Gaussian noise
    noise = tf.random.normal(shape=tf.shape(image), mean=0.0, stddev=0.02, dtype=tf.float32)
    image = image + noise
    image = tf.clip_by_value(image, 0.0, 1.0)

    # Cutout
    cutout_size = tf.random.uniform([], 30, 60, dtype=tf.int32)
    offset_height = tf.random.uniform([], 0, 256 - cutout_size, dtype=tf.int32)
    offset_width = tf.random.uniform([], 0, 256 - cutout_size, dtype=tf.int32)
    image = tf.tensor_scatter_nd_update(
        image,
        indices=tf.reshape(tf.range(offset_height, offset_height + cutout_size), (-1, 1)),
        updates=tf.zeros([cutout_size, 256, 3], dtype=tf.float32)
    )

    # Resize back
    image = tf.image.resize(image, (256, 256))
    mask = tf.image.resize(mask, (256, 256))

    return image, mask

class SkinDataset:
    def __init__(self, img_dir, mask_dir, img_size=(256, 256), batch_size=8, val_split=0.2):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size

        # Create dict of matchig files
        img_files = {f.rsplit(".", 1)[0]: os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith((".jpg", ".jpeg"))}
        mask_files = {f.rsplit(".", 1)[0]: os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")}

        #match the imgs with matching masks
        matched_filenames = sorted(set(img_files.keys()) & set(mask_files.keys()))

        img_paths = [img_files[f] for f in matched_filenames]
        mask_paths = [mask_files[f] for f in matched_filenames]

        #debug stuff
        if len(img_paths) != len(mask_paths):
            raise ValueError(f" Mismatch: {len(img_paths)} images, {len(mask_paths)} masks!")

        print(f" Found {len(img_paths)} matched image-mask pairs.")
        #-==================================

        #split to train and validation
        paired_data = list(zip(img_paths, mask_paths))
        train_data, val_data = train_test_split(paired_data, test_size=val_split, random_state=42)

        if train_data:
            self.train_img_paths, self.train_mask_paths = zip(*train_data)
            self.train_img_paths, self.train_mask_paths = list(self.train_img_paths), list(self.train_mask_paths)
        else:
            self.train_img_paths, self.train_mask_paths = [], []
            print(f" CAREFUL TRAIN IMG PATH IS EMPTY")

        if val_data:
            self.val_img_paths, self.val_mask_paths = zip(*val_data)
            self.val_img_paths, self.val_mask_paths = list(self.val_img_paths), list(self.val_mask_paths)
        else:
            self.val_img_paths, self.val_mask_paths = [], []
            print(f" CAREFUL TRAIN IMG PATH IS EMPTY")
            #dangerzone, so far never encounterred as of 2/26/2025

        print(f" Train: {len(self.train_img_paths)}, Validation: {len(self.val_img_paths)}")

    def load_image(self, img_path):
        """
        Load image

        """
        img_path = img_path.decode("utf-8")
        img = load_img(img_path, target_size=self.img_size)
        img = img_to_array(img) / 255.0
        return img.astype(np.float32)

    def load_mask(self, mask_path):
        mask_path = mask_path.decode("utf-8")
        mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        mask = np.where(mask > 0.5, 1.0, 0.0)  # Ensure binary masks
        #print("Unique mask values after binarization:", np.unique(mask))
        return mask.astype(np.float32)

    def get_train_val_datasets(self):
        train_dataset = self.to_tf_dataset(self.train_img_paths, self.train_mask_paths, augment=False)
        val_dataset = self.to_tf_dataset(self.val_img_paths, self.val_mask_paths, augment=False)
        return train_dataset, val_dataset

    def to_tf_dataset(self, image_paths, mask_paths, augment=False):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))
    
        def load_data(img_path, mask_path):
            img = tf.numpy_function(self.load_image, [img_path], tf.float32)
            mask = tf.numpy_function(self.load_mask, [mask_path], tf.float32)
    
            img.set_shape((256, 256, 3))
            mask.set_shape((256, 256, 1))
    
            return img, mask
    
        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
    
        if augment:
            dataset = dataset.map(lambda x, y: augment_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
    
        dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
        return dataset