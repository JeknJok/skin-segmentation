import tensorflow as tf
import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import cv2
import numpy as np

def augment_data(image, mask):
    combined = tf.concat([image, mask], axis=-1)
    combined = tf.image.random_flip_left_right(combined)  
    combined = tf.image.random_brightness(combined, max_delta=0.1)  
    image, mask = tf.split(combined, [3, 1], axis=-1)
    return image, mask


class SkinDataset:
    def __init__(self, img_dir, mask_dir, img_size=(256, 256), batch_size=8, val_split=0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size

        #self.image_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".jpeg")])
        #self.mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])
        self.train_img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".jpeg")])
        self.train_mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])

        # split trainind data to validation dataset too, using the val_split as ratio (val_split = 0.2 means 80% train 20% validation)
        #self.train_img_paths, self.val_img_paths, self.train_mask_paths, self.val_mask_paths = train_test_split(
            #self.image_paths, self.mask_paths, test_size=val_split, random_state=42)

    def load_image(self, img_path):
        img_path = img_path.decode("utf-8") 
        img = load_img(img_path, target_size=self.img_size)
        img = img_to_array(img) / 255.0
        return img.astype(np.float32)

    def load_mask(self, mask_path):
        mask_path = mask_path.decode("utf-8")
        mask = load_img(mask_path, target_size=self.img_size, color_mode="grayscale")
        mask = img_to_array(mask) / 255.0  # Normalize to [0,1] range
        mask = (mask > 0.5).astype("float32")  # Ensure binary mask
        return mask

    
    def get_train_val_datasets(self):
        train_dataset = self.to_tf_dataset(self.train_img_paths, self.train_mask_paths)
        #val_dataset = self.to_tf_dataset(self.val_img_paths, self.val_mask_paths)
        #return train_dataset, val_dataset
        return train_dataset
    
    def to_tf_dataset(self, image_paths, mask_paths, augment=True):
        dataset = tf.data.Dataset.from_tensor_slices((image_paths, mask_paths))

        def load_data(img_path, mask_path):
            img = tf.numpy_function(self.load_image, [img_path], tf.float32)
            mask = tf.numpy_function(self.load_mask, [mask_path], tf.float32)

            img.set_shape((self.img_size[0], self.img_size[1], 3))  
            mask.set_shape((self.img_size[0], self.img_size[1], 1))  

            return img, mask

        dataset = dataset.map(load_data, num_parallel_calls=tf.data.AUTOTUNE)
        
        if augment:
            #dataset = dataset.map(lambda x, y: augment_data(x, y), num_parallel_calls=tf.data.AUTOTUNE)
            dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)  # No Augmentation

        dataset = dataset.batch(self.batch_size)  # Apply batching here ONCE
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset