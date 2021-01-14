import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras import losses
# tf.config.gpu.set_per_process_memory_fraction(0.5)

def dice_coeff(y_true, y_pred):
    smooth = 1.
    # Flatten
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f ** 2) + tf.reduce_sum(y_pred_f ** 2) + smooth)
    # score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    pt_1 = tf.where(tf.equal(y_true, 1), y_pred, tf.ones_like(y_pred))
    pt_0 = tf.where(tf.equal(y_true, 0), y_pred, tf.zeros_like(y_pred))
    epsilon = K.epsilon()
    # clip to prevent NaN's and Inf's
    pt_1 = K.clip(pt_1, epsilon, 1. - epsilon)
    pt_0 = K.clip(pt_0, epsilon, 1. - epsilon)
    loss = -K.sum(alpha * K.pow(1. - pt_1, gamma) * K.log(pt_1)) \
            -K.sum((1 - alpha) * K.pow(pt_0, gamma) * K.log(1. - pt_0))
    loss = loss/tf.cast(tf.size(y_true), tf.float32)
    return loss

def bce_dice_loss(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    loss = 20*losses.binary_crossentropy(y_true, y_pred) + dice_loss(y_true, y_pred)
    # loss = dice_loss(y_true, y_pred)
    # loss = 100 * binary_focal_loss(y_true, y_pred, 1,0.25) + dice_loss(y_true, y_pred)
    return loss

def get_shallow_unet(size=None, t=32): #(size)
    inputs = tf.keras.layers.Input((size, size, 1))
    # inputs = tf.keras.layers.Input((None, None, 1))
    # s = tf.keras.layers.Lambda(lambda x: x / 255)(inputs)

    c1 = tf.keras.layers.Conv2D(t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(inputs)
    c1 = tf.keras.layers.Conv2D(t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c1)
    # c1 = tf.keras.layers.Dropout(0.25)(c1)
    p1 = tf.keras.layers.MaxPooling2D((2, 2))(c1)

    c2 = tf.keras.layers.Conv2D(2*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p1)
    c2 = tf.keras.layers.Conv2D(2*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c2)
    c2 = tf.keras.layers.Dropout(0.1)(c2)
    #c2 = tf.keras.layers.Conv2D(8, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c2)
    p2 = tf.keras.layers.MaxPooling2D((2, 2))(c2)

    c3 = tf.keras.layers.Conv2D(4*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(p2)
    c3 = tf.keras.layers.Conv2D(4*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(c3)
    c3 = tf.keras.layers.Dropout(0.2)(c3)
    #c3 = tf.keras.layers.Conv2D(16, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
    #                            padding='same')(c3)
    p3 = tf.keras.layers.MaxPooling2D((2, 2))(c3)

    c4 = tf.keras.layers.Conv2D(8*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(p3)
    c4 = tf.keras.layers.Conv2D(8*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c4)
    c4 = tf.keras.layers.Dropout(0.2)(c4)
    p4 = tf.keras.layers.MaxPooling2D(pool_size=(2, 2))(c4)

    c5 = tf.keras.layers.Conv2D(16*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(p4)
    c5 = tf.keras.layers.Conv2D(16*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c5)
    c5 = tf.keras.layers.Dropout(0.2)(c5)

    u6 = tf.keras.layers.Conv2DTranspose(8*t, (2, 2), strides=(2, 2), padding='same')(c5)
    u6 = tf.keras.layers.concatenate([u6, c4], axis=3)

    c6 = tf.keras.layers.Conv2D(8*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(u6)
    c6 = tf.keras.layers.Conv2D(256, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c6)
    c6 = tf.keras.layers.Dropout(0.2)(c6)

    u7 = tf.keras.layers.Conv2DTranspose(4*t, (2, 2), strides=(2, 2), padding='same')(c6)
    u7 = tf.keras.layers.concatenate([u7, c3], axis=3)
    c7 = tf.keras.layers.Conv2D(4*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u7)
    c7 = tf.keras.layers.Conv2D(4*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c7)
    c7 = tf.keras.layers.Dropout(0.2)(c7)

    u8 = tf.keras.layers.Conv2DTranspose(2*t, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = tf.keras.layers.concatenate([u8, c2], axis=3)
    c8 = tf.keras.layers.Conv2D(2*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u8)
    c8 = tf.keras.layers.Conv2D(2*t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c8)
    c8 = tf.keras.layers.Dropout(0.1)(c8)

    u9 = tf.keras.layers.Conv2DTranspose(t, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = tf.keras.layers.concatenate([u9, c1], axis=3)
    c9 = tf.keras.layers.Conv2D(t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                                padding='same')(u9)
    c9 = tf.keras.layers.Conv2D(t, (3, 3), activation=tf.keras.activations.elu, kernel_initializer='he_normal',
                               padding='same')(c9)
    # c9 = tf.keras.layers.Dropout(0.1)(c9)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(c9) # sigmoid, softmax

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    # model.compile(optimizer='adam', loss=dice_loss, metrics=[dice_loss])
    model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_loss])
    # model.summary()
    return model

if __name__ == '__main__':
    model = get_unet()
    model.summary()
