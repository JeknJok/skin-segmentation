import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
print(tf.__version__)
print(tf.config.list_physical_devices('GPU')) 
print("TensorFlow Version:", tf.__version__)
print("Available GPUs:", tf.config.list_physical_devices('GPU'))
print("Built with CUDA:", tf.test.is_built_with_cuda())
 
#Create a simple computation on GPU
with tf.device('/GPU:0'):
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print("Matrix multiplication result:\n", c.numpy())

    