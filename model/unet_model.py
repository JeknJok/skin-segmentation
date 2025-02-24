import tensorflow as tf

#tversky loss https://www.tensorflow.org/api_docs/python/tf/keras/losses/tversky
@tf.keras.utils.register_keras_serializable()
def tversky_loss(y_true, y_pred, alpha=0.25, beta=0.75, smooth=1):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    true_pos = tf.keras.backend.sum(y_true_f * y_pred_f)
    false_neg = tf.keras.backend.sum(y_true_f * (1 - y_pred_f))
    false_pos = tf.keras.backend.sum((1 - y_true_f) * y_pred_f)
    return 1 - (true_pos + smooth) / (true_pos + alpha * false_neg + beta * false_pos + smooth)

@tf.keras.utils.register_keras_serializable()
def focal_tversky_loss(y_true, y_pred, alpha=0.5, beta=0.5, gamma=1.2, smooth=1):
    tversky = tversky_loss(y_true, y_pred, alpha, beta, smooth)
    return tf.keras.backend.pow((1 - tversky), gamma)

#adapted from: https://stackoverflow.com/questions/72195156/correct-implementation-of-dice-loss-in-tensorflow-keras
# use dice loss instead of binary cross-entropy loss.
@tf.keras.utils.register_keras_serializable()
def dice_coef(y_true, y_pred, smooth=1.0,beta=0.7):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    intersection = tf.keras.backend.sum(y_true_f * y_pred_f)
    return 1 - ((2. * intersection + smooth) / 
                (beta * tf.keras.backend.sum(y_true_f) + 
                 (1 - beta) * tf.keras.backend.sum(y_pred_f) + smooth))

# register the dice loss
@tf.keras.utils.register_keras_serializable()
def dice_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

# adapted from: https://github.com/artemmavrin/focal-loss/blob/master/src/focal_loss/_binary_focal_loss.py
#adding focal loss to combine with dice loss
@tf.keras.utils.register_keras_serializable()
def focal_loss(y_true, y_pred, alpha=0.5, gamma=1.5):
    y_true_f = tf.keras.backend.flatten(y_true)
    y_pred_f = tf.keras.backend.flatten(y_pred)
    binarycrossentropy = tf.keras.backend.binary_crossentropy(y_true_f, y_pred_f)
    loss = alpha * tf.keras.backend.pow((1 - y_pred_f), gamma) * binarycrossentropy
    return tf.keras.backend.mean(loss)

@tf.keras.utils.register_keras_serializable()
def combined_loss(y_true, y_pred):
    return 0.7 * dice_loss(y_true, y_pred) + 0.3 * focal_loss(y_true, y_pred)

#sharper detection
def attention_block(x, gating, num_filters):
    """ Attention mechanism to refine feature selection. """
    x = tf.keras.layers.Conv2D(num_filters, (1, 1), padding="same")(x)
    gating = tf.keras.layers.Conv2D(num_filters, (1, 1), padding="same")(gating)
    add = tf.keras.layers.Add()([x, gating])
    add = tf.keras.layers.Activation("relu")(add)
    attention = tf.keras.layers.Conv2D(1, (1, 1), activation="sigmoid")(add)
    return tf.keras.layers.Multiply()([x, attention])

# This U-NET ARCHITECTURE code is adapted from: https://www.geeksforgeeks.org/u-net-architecture-explained/
def encoder_block(inputs, num_filters):
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)  
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    p = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """ Decoder block with attention mechanism. """
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = attention_block(x, skip_features, num_filters)  #attention layer
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    return x

def unet_model(input_shape=(256, 256, 3), num_classes=1): 
    inputs = tf.keras.layers.Input(input_shape)

    # Contracting Path (Encoder)
    s1, p1 = encoder_block(inputs, 64)
    s2, p2 = encoder_block(p1, 128) 
    s3, p3 = encoder_block(p2, 256) 
    s4, p4 = encoder_block(p3, 512)

    # Bottleneck
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same')(p4) 
    b1 = tf.keras.layers.Activation('relu')(b1) 
    b1 = tf.keras.layers.Conv2D(1024, 3, padding='same')(b1) 
    b1 = tf.keras.layers.Activation('relu')(b1) 

    # Expansive Path (Decoder)
    d1 = decoder_block(b1, s4, 512) 
    d2 = decoder_block(d1, s3, 256) 
    d3 = decoder_block(d2, s2, 128) 
    d4 = decoder_block(d3, s1, 64) 

    # Output Layer
    outputs = tf.keras.layers.Conv2D(1, 1, activation='relu', padding='same')(d4)
    outputs = tf.keras.layers.Lambda(lambda x: tf.clip_by_value(x, 0, 1))(outputs)

    model = tf.keras.models.Model(inputs, outputs, name='U-Net') 
    return model