import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.layers import Conv2D, BatchNormalization, ReLU, MaxPooling2D, UpSampling2D, Concatenate, Input
from tensorflow.keras.saving import register_keras_serializable
import tensorflow.keras.backend as K

@register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.7, beta=0.3, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    true_pos = K.sum(y_true * y_pred)
    false_neg = K.sum(y_true * (1 - y_pred))
    false_pos = K.sum((1 - y_true) * y_pred)
    return 1 - ((true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth))

@register_keras_serializable()
def weighted_binary_crossentropy(y_true, y_pred, pos_weight=2.0, smooth=1e-6):
    y_pred = tf.clip_by_value(y_pred, smooth, 1 - smooth)
    loss = - (pos_weight * y_true * K.log(y_pred) + (1 - y_true) * K.log(1 - y_pred))
    return K.mean(loss)

@register_keras_serializable()
def mean_iou(y_true, y_pred, smooth=1e-6):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    intersection = K.sum(y_true * y_pred)
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)

@register_keras_serializable()
def iou_loss(targets, inputs, smooth=1e-6):
    inputs = K.flatten(inputs)
    targets = K.flatten(targets)
    intersection = K.sum(targets * inputs)
    union = K.sum(targets) + K.sum(inputs) - intersection
    iou = (intersection + smooth) / (union + smooth)
    return 1 - iou

@register_keras_serializable()
def dice_loss(y_true, y_pred, smooth=1):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth))

@register_keras_serializable()
def combined_loss(y_true, y_pred):
    return (
        0.3 * weighted_binary_crossentropy(y_true, y_pred) +
        0.3 * tversky_loss(y_true, y_pred) +
        0.4 * iou_loss(y_true, y_pred)
    )

# --- U2NET Components ---
def REBNCONV(x, out_ch, dilation=1, name=None):
    x = Conv2D(out_ch, 3, padding='same', dilation_rate=dilation, name=f'{name}_conv')(x)
    x = BatchNormalization(name=f'{name}_bn')(x)
    x = ReLU(name=f'{name}_relu')(x)
    return x

# --- RSU Blocks ---
def RSU_block(x, in_ch, mid_ch, out_ch, depth, name, use_dilation=False):
    hx_in = REBNCONV(x, out_ch, name=f'{name}_in')
    skips = []
    hx = hx_in
    # Encoder
    for d in range(1, depth):
        hx = REBNCONV(hx, mid_ch, name=f'{name}_conv{d}')
        skips.append(hx)
        hx = MaxPooling2D(2, strides=2, padding='same')(hx)
    # Bottom
    hx = REBNCONV(hx, mid_ch, dilation=2 if use_dilation else 1, name=f'{name}_conv{depth}')
    # Decoder
    for d in reversed(range(1, depth)):
        hx = UpSampling2D(size=2, interpolation='bilinear')(hx)
        hx = REBNCONV(Concatenate()([hx, skips[d-1]]), mid_ch, name=f'{name}_up{d}')
    # Final
    hx = REBNCONV(Concatenate()([hx, skips[0]]), out_ch, name=f'{name}_up0')
    return tf.keras.layers.Add()([hx, hx_in])

def RSU7(x, in_ch, mid_ch, out_ch, name):
    return RSU_block(x, in_ch, mid_ch, out_ch, depth=7, name=name)

def RSU6(x, in_ch, mid_ch, out_ch, name):
    return RSU_block(x, in_ch, mid_ch, out_ch, depth=6, name=name)

def RSU5(x, in_ch, mid_ch, out_ch, name):
    return RSU_block(x, in_ch, mid_ch, out_ch, depth=5, name=name)

def RSU4(x, in_ch, mid_ch, out_ch, name):
    return RSU_block(x, in_ch, mid_ch, out_ch, depth=4, name=name)

def RSU4F(x, in_ch, mid_ch, out_ch, name):
    hx_in = REBNCONV(x, out_ch, name=f'{name}_in')
    hx1 = REBNCONV(hx_in, mid_ch, name=f'{name}_conv1')
    hx2 = REBNCONV(hx1, mid_ch, dilation=2, name=f'{name}_conv2')
    hx3 = REBNCONV(hx2, mid_ch, dilation=4, name=f'{name}_conv3')
    hx4 = REBNCONV(hx3, mid_ch, dilation=8, name=f'{name}_conv4')
    hx3d = REBNCONV(Concatenate()([hx4, hx3]), mid_ch, dilation=4, name=f'{name}_up3')
    hx2d = REBNCONV(Concatenate()([hx3d, hx2]), mid_ch, dilation=2, name=f'{name}_up2')
    hx1d = REBNCONV(Concatenate()([hx2d, hx1]), out_ch, dilation=1, name=f'{name}_up1')
    return tf.keras.layers.Add()([hx1d, hx_in])

# --- Full U2NET Model ---
def model(input_shape=(256, 256, 3), num_classes=1):
    inputs = Input(shape=input_shape)

    stage1 = RSU7(inputs, 3, 32, 64, name='stage1')
    pool1 = MaxPooling2D(2, strides=2, padding='same')(stage1)

    stage2 = RSU6(pool1, 64, 32, 128, name='stage2')
    pool2 = MaxPooling2D(2, strides=2, padding='same')(stage2)

    stage3 = RSU5(pool2, 128, 64, 256, name='stage3')
    pool3 = MaxPooling2D(2, strides=2, padding='same')(stage3)

    stage4 = RSU4(pool3, 256, 128, 512, name='stage4')
    pool4 = MaxPooling2D(2, strides=2, padding='same')(stage4)

    stage5 = RSU4F(pool4, 512, 256, 512, name='stage5')
    pool5 = MaxPooling2D(2, strides=2, padding='same')(stage5)

    stage6 = RSU4F(pool5, 512, 256, 512, name='stage6')

    # Decoder
    stage5d = RSU4F(Concatenate()([UpSampling2D(size=2, interpolation='bilinear')(stage6), stage5]), 1024, 256, 512, name='stage5d')
    stage4d = RSU4(Concatenate()([UpSampling2D(size=2, interpolation='bilinear')(stage5d), stage4]), 1024, 128, 256, name='stage4d')
    stage3d = RSU5(Concatenate()([UpSampling2D(size=2, interpolation='bilinear')(stage4d), stage3]), 512, 64, 128, name='stage3d')
    stage2d = RSU6(Concatenate()([UpSampling2D(size=2, interpolation='bilinear')(stage3d), stage2]), 256, 32, 64, name='stage2d')
    stage1d = RSU7(Concatenate()([UpSampling2D(size=2, interpolation='bilinear')(stage2d), stage1]), 128, 16, 64, name='stage1d')

    outputs = Conv2D(num_classes, 1, activation='sigmoid', name='final_output')(stage1d)
    model = Model(inputs, outputs, name="U2NET_Keras")
    return model
