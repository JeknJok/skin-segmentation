import tensorflow as tf
import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np

def augment_data(image, mask):
    """
    Perform data augmentation on the image and mask together.
    The mask and image are first combined into one, and are given random modifications.
    Then they are split back to two and are returned.
    
    Input: mask, image
    Output, mask, image, the same ones at the same time.
    
    """
    combined = tf.concat([image, mask], axis=-1) 
    combined = tf.image.random_flip_left_right(combined)
    combined = tf.image.random_flip_up_down(combined)
    combined = tf.image.rot90(combined, k=tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))
    image, mask = tf.split(combined, [3, 1], axis=-1)
    return image, mask

class SkinDataset:
    def __init__(self, img_dir, mask_dir, img_size=(256, 256), batch_size=8, val_split=0):
        self.img_dir = img_dir
        self.mask_dir = mask_dir
        self.img_size = img_size
        self.batch_size = batch_size
        
        self.train_img_paths = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith(".jpg") or f.endswith(".jpeg")])
        self.train_mask_paths = sorted([os.path.join(mask_dir, f) for f in os.listdir(mask_dir) if f.endswith(".png")])


    def load_image(self, img_path):
        """ 
        Load image
         
        """
        img_path = img_path.decode("utf-8") 
        img = load_img(img_path, target_size=self.img_size)
        img = img_to_array(img) / 255.0
        return img.astype(np.float32)
    
    def load_mask(self, mask_path):
        """ 
        Load and preprocess a PNG mask 

        """
        mask_path = mask_path.decode("utf-8")
        mask = load_img(mask_path, target_size=(256, 256), color_mode="grayscale")
        mask = img_to_array(mask) / 255.0
        return mask.astype(np.float32)

    
    def get_train_val_datasets(self):
        train_dataset = self.to_tf_dataset(self.train_img_paths, self.train_mask_paths)
        return train_dataset
    
    def to_tf_dataset(self, image_paths, mask_paths, augment=True):
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
            #dataset = dataset.map(lambda x, y: (x, y), num_parallel_calls=tf.data.AUTOTUNE)

        dataset = dataset.batch(self.batch_size)
        dataset = dataset.prefetch(tf.data.AUTOTUNE)
        return dataset