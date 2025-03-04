
## dataset.py
import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from sklearn.model_selection import train_test_split

def augment_data(image, mask):
    """
    Perform data augmentation on the image and mask together.
    The mask and image are first combined into one, and are given random modifications.
    Then they are split back to two and are returned.

    Input: mask, image
    Output, mask, image, the same ones at the same time.

    """
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.clip_by_value(image, 0.0, 1.0)

    combined = tf.concat([image, mask], axis=-1)
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    combined = tf.image.rot90(combined, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, mask = tf.split(combined, [3, 1], axis=-1)

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
        """
        dataset returned as tf dataset.
        """
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

        def load_data(img_path, mask_path):
            img = tf.numpy_function(self.load_image, [img_path], tf.float32)
            mask = tf.numpy_function(self.load_mask, [mask_path], tf.float32)

            img.set_shape((self.img_size[0], self.img_size[1], 3))
            mask.set_shape((self.img_size[0], self.img_size[1], 1))

            return img, mask

        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)

        if augment:
            dataset = dataset.map(lambda x, y: augment_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)
        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset