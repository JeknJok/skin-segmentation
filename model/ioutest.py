import tensorflow as tf
import numpy as np
from unet_model import model, iou_loss, iou_loss, mean_iou,combined_loss

# 
y_true_sample = np.random.randint(0, 2, (2, 256, 256, 1)).astype(np.float32)
y_pred_sample = np.random.rand(2, 256, 256, 1).astype(np.float32)  

# Convert to tensors
y_true_sample = tf.convert_to_tensor(y_true_sample)
y_pred_sample = tf.convert_to_tensor(y_pred_sample)

# Compute loss
loss_value = iou_loss(y_true_sample, y_pred_sample)
print("Sample IoU Loss:", loss_value.numpy())  # Should be between 0 and 1