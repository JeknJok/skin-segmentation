import tensorflow as tf

# This U-NET ARCHITECTURE code is adapted from: https://www.geeksforgeeks.org/u-net-architecture-explained/

def encoder_block(inputs, num_filters, dropout_rate=0.2):
    """Encoder block: Conv2D -> BatchNorm -> ReLU -> Dropout -> Conv2D -> BatchNorm -> ReLU -> MaxPooling"""
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(inputs)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    x = tf.keras.layers.Dropout(dropout_rate)(x)  # New dropout layer
    x = tf.keras.layers.Conv2D(num_filters, 3, padding='same')(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Activation('relu')(x)
    p = tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2)(x)
    return x, p

def decoder_block(inputs, skip_features, num_filters):
    """Decoder block: Upsample -> Concatenate -> Conv2D -> BatchNorm -> ReLU -> Conv2D -> BatchNorm -> ReLU"""
    x = tf.keras.layers.Conv2DTranspose(num_filters, (2, 2), strides=2, padding='same')(inputs)
    x = tf.keras.layers.Concatenate()([x, skip_features])
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
    outputs = tf.keras.layers.Conv2D(num_classes, 1, padding='same', activation='sigmoid')(d4) 

    model = tf.keras.models.Model(inputs, outputs, name='U-Net') 
    return model