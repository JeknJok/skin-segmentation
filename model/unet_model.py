import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

# custom IOU metric function
@register_keras_serializable()
def mean_iou(y_true, y_pred, smooth=1e-6):
    """
    labels,prediction with shape of [batch,height,width,class_number=2]
    this code is adapted from: https://stackoverflow.com/questions/49715192/tensorflow-mean-iou-for-just-foreground-class-for-binary-semantic-segmentation
    updated by me to work with TensorFlow 2.x + Keras:

    Args:
        y_true: Ground truth mask (batch, height, width, 1)
        y_pred: Predicted mask (batch, height, width, 1)

    Returns:
        Mean IoU score

    """
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)

    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def iou_loss(targets, inputs, smooth=1e-6):
    """
    from https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch
    adapted for the binary segmentation nature of this project
    """
    #flatten tensors
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)

    intersection = K.sum(targets * inputs)
    union = K.sum(targets) + K.sum(inputs) - intersection
    iou = (intersection + smooth) / (union + smooth)
    
    return 1 - iou

@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)  # Prevents log(0) issues
    intersection = K.sum(y_true * y_pred)
    return 1 - (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_pred) + smooth)
    
@register_keras_serializable()
def combined_loss(y_true, y_pred):
    return (
        0.5 * tf.keras.losses.BinaryCrossentropy()(y_true, y_pred) +
        #x.x * dice_loss(y_true, y_pred) +
        0.5 * iou_loss(y_true, y_pred)
    )

def model(input_shape=(256, 256, 3), num_classes=1):

    """
    Using ResNet50 pretrained model as encoder block of the UNET architechture of this model.
    ResNet50 being finetuned, freeze the first x (as seen in program, subject to changes) layers
    essentially, the first x layers are "frozen" and we only use those that are influential to better learn through the dataset.
    Since ResNet50 Convolutional Blocks (for downsampling) and Identity Blocks (for residual connections), I am simply just leaving these
    features as is and attaching A U-NET decoder.

    # U-NET architechture adapted from: https://www.geeksforgeeks.org/u-net-architecture-explained/
    # RESNET50: Adapted from https://colab.research.google.com/github/yashclone999/ResNet_MODEL/blob/master/ResNet50.ipynb#scrollTo=Hb0_EeKk-DeH

    """
    # Load ResNet50 as encoder (without top layers)
    local_weights_path = "/kaggle/input/resnet50_weights/tensorflow2/default/1/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5"
    base_model = ResNet50(weights=None, include_top=False, input_shape=input_shape, name="resnet50")
    base_model.load_weights(local_weights_path)

    #finetuning, freeze all layers
    for layer in base_model.layers:
        layer.trainable = False

    #encoder layers (skip connections)
    skip1 = base_model.get_layer("conv1_relu").output
    skip2 = base_model.get_layer("conv2_block3_out").output
    skip3 = base_model.get_layer("conv3_block4_out").output
    skip4 = base_model.get_layer("conv4_block6_out").output

    bottleneck = base_model.get_layer("conv5_block3_out").output

    # decoder
    up1 = layers.Conv2DTranspose(512, (3, 3), strides=(2, 2), padding="same")(bottleneck)
    up1 = layers.Concatenate()([up1, skip4])
    up1 = layers.Conv2D(512, 3, activation="relu", padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001))(up1)
    up1 = layers.Dropout(0.4)(up1)
    up1 = layers.BatchNormalization(momentum=0.8)(up1)
    
    up2 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(up1)
    up2 = layers.Concatenate()([up2, skip3])
    up2 = layers.Conv2D(256, 3, activation="relu", padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001))(up2)
    up2 = layers.Dropout(0.4)(up2)
    up2 = layers.BatchNormalization(momentum=0.8)(up2)
    

    up3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(up2)
    up3 = layers.Concatenate()([up3, skip2])
    up3 = layers.Conv2D(128, 3, activation="relu", padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001))(up3)
    up3 = layers.Dropout(0.4)(up3)
    up3 = layers.BatchNormalization(momentum=0.8)(up3)
    

    up4 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(up3)
    up4 = layers.Concatenate()([up4, skip1])
    up4 = layers.Conv2D(128, 3, activation="relu", padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001))(up4)
    up4 = layers.Dropout(0.4)(up4)
    up4 = layers.BatchNormalization(momentum=0.8)(up4)

    #upsampling to 256x256
    up5 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(up4)
    up5 = layers.Conv2D(64, 3, activation="relu", padding="same",kernel_regularizer=tf.keras.regularizers.l2(0.001))(up5)

    #
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(up5)

    model = Model(inputs=base_model.input, outputs=outputs, name="U-Net_ResNet50")

    return model
