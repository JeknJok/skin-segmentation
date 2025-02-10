import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array

def augment_data(image, mask):
        """
        Perform data augmentation on the image and mask, to improve the quality of training dataset
        """
        #Combine image and mask for consistent augmentation
        combined = tf.concat([image, mask], axis=-1)
        #random horizontal flip
        combined = tf.image.random_flip_left_right(combined)
        #random brightness
        image = tf.image.random_brightness(image, max_delta=0.1)
        #separate augmented image and mask
        image, mask = tf.split(combined, [3, 1], axis=-1)
        return image, mask

class SkinDataset:
    def __init__(self, img_dir, mask_dir, img_size=(256, 256), batch_size=8, val_split=0.2):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size

        self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".jpeg")])
        self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])  

        # Split into training and validation sets
        self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = train_test_split(
            self.image_paths, self.mask_paths, test_size=val_split, random_state=42)

    def load_image(self, img_path):
        """ Load and preprocess 1 image """
        img_path = img_path.decode("utf-8")  # Decode TensorFlow tensor path
        img = load_img(img_path, target_size=self.img_size)
        img = img_to_array(img) / 255.0  # Normalize to [0,1]
        return img.astype(np.float32)

    def load_mask(self, mask_path):
        """ Load and preprocess a PNG mask """
        mask_path = mask_path.decode("utf-8")  # Decode TensorFlow tensor path
        mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")  # Load as grayscale
        mask = img_to_array(mask) / 255.0  # Normalize to [0,1]
        return mask.astype(np.float32)
    
    def get_train_val_datasets(self):
        """ Get both training and validation datasets """
        train_dataset = self.to_tf_dataset(self.train_img_paths, self.train_mask_paths)
        val_dataset = self.to_tf_dataset(self.val_img_paths, self.val_mask_paths)
        return train_dataset, val_dataset
    
    def to_tf_dataset(self, image_paths, mask_paths, augment=True):
            """ Convert dataset to TensorFlow `tf.data.Dataset` """
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
            dataset = dataset.batch(self.batch_size).prefetch(tf.data.AUTOTUNE)
            return dataset