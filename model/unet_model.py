import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.saving import register_keras_serializable

#to make predictions match ground truth
@register_keras_serializable()
def loss_rescale(y_true, y_pred):
    y_pred_resized = tf.image.resize(y_pred, (256, 256))  
    return tf.keras.losses.BinaryCrossentropy()(y_true, y_pred_resized)

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
    base_model = ResNet50(weights="imagenet", include_top=False, input_shape=input_shape, name="resnet50")

    #finetuning
    for layer in base_model.layers[:100]:
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
    up1 = layers.Conv2D(512, 3, activation="relu", padding="same")(up1)

    up2 = layers.Conv2DTranspose(256, (3, 3), strides=(2, 2), padding="same")(up1)
    up2 = layers.Concatenate()([up2, skip3])
    up2 = layers.Conv2D(256, 3, activation="relu", padding="same")(up2)

    up3 = layers.Conv2DTranspose(128, (3, 3), strides=(2, 2), padding="same")(up2)
    up3 = layers.Concatenate()([up3, skip2])
    up3 = layers.Conv2D(128, 3, activation="relu", padding="same")(up3)

    up4 = layers.Conv2DTranspose(64, (3, 3), strides=(2, 2), padding="same")(up3)
    up4 = layers.Concatenate()([up4, skip1])
    up4 = layers.Conv2D(64, 3, activation="relu", padding="same")(up4)

    #upsampling to 256x256
    up5 = layers.Conv2DTranspose(32, (3, 3), strides=(2, 2), padding="same")(up4)
    up5 = layers.Conv2D(32, 3, activation="relu", padding="same")(up5)

    #
    outputs = layers.Conv2D(1, (1, 1), activation="sigmoid")(up5)

    model = Model(inputs=base_model.input, outputs=outputs, name="U-Net_ResNet50")

    return model

