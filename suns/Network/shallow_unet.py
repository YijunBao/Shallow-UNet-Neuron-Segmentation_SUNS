import tensorflow as tf
from keras import backend as K
from tensorflow.python.keras import losses


def dice_coeff(y_true, y_pred):
    '''Dice coefficient between two arrays.

    Inputs: 
        y_true (tf.TensorArray): GT array
        y_pred (tf.TensorArray): Predicted array by CNN

    Outputs:
        score (tf.float32): dice coefficient. 
    '''
    smooth = 1.
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f ** 2) + tf.reduce_sum(y_pred_f ** 2) + smooth)
    return score

def dice_loss(y_true, y_pred):
    '''Dice loss between two arrays.

    Inputs: 
        y_true (tf.TensorArray): GT array
        y_pred (tf.TensorArray): Predicted array by CNN

    Outputs:
        loss (tf.float32): dice loss. 
    '''
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss

def binary_focal_loss(y_true, y_pred, gamma=2., alpha=0.25):
    '''Binary focal loss between two arrays.

    Inputs: 
        y_true (tf.TensorArray): GT array
        y_pred (tf.TensorArray): Predicted array by CNN
        gamma (float, default to 2): first parameter of focal loss
        alpha (float, default to 0.25): second parameter of focal loss

    Outputs:
        loss (tf.float32): binary focal loss. 
    '''
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

def total_loss(y_true, y_pred, DL=1, BCE=20, FL=0, gamma=1, alpha=0.25):
    '''Total loss between two arrays. Can be linear superposition of multiple loss functions,
        including dice_loss, binary_corss_entropy, and focal_loss. 

    Inputs: 
        y_true (tf.TensorArray): GT array
        y_pred (tf.TensorArray): Predicted array by CNN
        DL(float, default to 1): Coefficient of dice loss in the total loss
        BCE(float, default to 20): Coefficient of binary cross entropy in the total loss
        FL(float, default to 0): Coefficient of focal loss in the total loss
        gamma (float, default to 1): first parameter of focal loss
        alpha (float, default to 0.25): second parameter of focal loss

    Outputs:
        loss (tf.float32): binary focal loss. 
    '''
    y_true = tf.cast(y_true, tf.float32)
    loss = DL * dice_loss(y_true, y_pred)
    if BCE:
        loss += BCE * losses.binary_crossentropy(y_true, y_pred)
    if DL:
        loss += FL * binary_focal_loss(y_true, y_pred, gamma, alpha)
    return loss


def get_shallow_unet(size=None, Params_loss=None):
    '''Get a shallow U-Net model. This is the optimal shallow U-Net after our test.
        In training, "Params_loss" specifies the loss function.

    Inputs: 
        size (int, default to None): Lateral size of each dimension of the input layer.
            Usually use None is sufficient, and using None can only cover rectangular input.  
        Params_loss(dict, default to None): parameters of the loss function "total_loss". Only used in training.
            Params_loss['DL'](float): Coefficient of dice loss in the total loss
            Params_loss['BCE'](float): Coefficient of binary cross entropy in the total loss
            Params_loss['FL'](float): Coefficient of focal loss in the total loss
            Params_loss['gamma'] (float): first parameter of focal loss
            Params_loss['alpha'] (float): second parameter of focal loss

    Outputs:
        model: the CNN model. 
    '''
    activation=tf.keras.activations.elu
    # activation=tf.keras.activations.relu
    inputs = tf.keras.layers.Input((size, size, 1))

    c1 = tf.keras.layers.Conv2D(4, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(inputs)
    do1 = tf.keras.layers.Dropout(0.1)(c1)

    mp1 = tf.keras.layers.MaxPooling2D((2, 2))(do1)
    c2 = tf.keras.layers.Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp1)
    do2 = tf.keras.layers.Dropout(0.1)(c2)

    mp2 = tf.keras.layers.MaxPooling2D((2, 2))(do2)
    c3 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp2)
    do3 = tf.keras.layers.Dropout(0.2)(c3)

    c3 = tf.keras.layers.Conv2D(16, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do3)
    do3 = tf.keras.layers.Dropout(0.2)(c3)

    ct2 = tf.keras.layers.Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same')(do3)
    # ct2 = tf.keras.layers.concatenate([ct2, do2], axis=3) # The second skip connection

    c2 = tf.keras.layers.Conv2D(8, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct2)
    do2 = tf.keras.layers.Dropout(0.1)(c2)

    ct1 = tf.keras.layers.Conv2DTranspose(4, (2, 2), strides=(2, 2), padding='same')(do2)
    ct1 = tf.keras.layers.concatenate([ct1, do1], axis=3) # The first skip connection

    c1 = tf.keras.layers.Conv2D(4, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct1)
    do1 = tf.keras.layers.Dropout(0.1)(c1)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(do1)

    def loss_func(y_true, y_pred):
        if Params_loss is None:
            return total_loss(y_true, y_pred)
        else:
            return total_loss(y_true, y_pred, DL=Params_loss['DL'], BCE=Params_loss['BCE'], \
                FL=Params_loss['FL'], gamma=Params_loss['gamma'], alpha=Params_loss['alpha'])

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_func, metrics=[dice_loss])
    # model.compile(optimizer=tf.keras.optimizers.Adam(lr=1e-4), loss=loss_func, metrics=[dice_loss])
    return model


def get_shallow_unet_more(size=None, n_depth=3, n_channel=4, skip=[1], activation='elu', Params_loss=None):
    '''Get a shallow U-Net model. This function has some options to vary the model, 
        including the number of resolution depth ("n_depth"), number of channels per featrue map ("n_channel"), 
        number of skip connections ("skip"), and choice of activation function ("activation").
        The number of channels are doubled on each depth. 

    Inputs: 
        size (int, default to None): Lateral size of each dimension of the input layer.
            Usually using None is sufficient, and using None can only cover rectangular input. 
        n_depth (int, default to 3): Number of resolution depth. Can be 2, 3, or 4
        n_channel (int, default to 4): Number of channels per feature map in the first resolution depth.
            For deeper depths, the numbers of feature map will double for every depth. 
        skip (list of int, default to [1]]): Indeces of resolution depths that use skip connections.
            The shallowest depth is 1, and 1 should usually be in "skip".
        activation (str, default to 'elu): activation function. Can be 'elu' or 'relu'.
        Params_loss(dict, default to None): parameters of the loss function "total_loss". Only used in training.
            Params_loss['DL'](float): Coefficient of dice loss in the total loss
            Params_loss['BCE'](float): Coefficient of binary cross entropy in the total loss
            Params_loss['FL'](float): Coefficient of focal loss in the total loss
            Params_loss['gamma'] (float): first parameter of focal loss
            Params_loss['alpha'] (float): second parameter of focal loss

    Outputs:
        model: the CNN model. 
    '''
    if activation=='elu':
        activation=tf.keras.activations.elu
    elif activation=='relu':
        activation=tf.keras.activations.relu
    inputs = tf.keras.layers.Input((size, size, 1))

    c1 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(inputs)
    do1 = tf.keras.layers.Dropout(0.1)(c1)

    mp1 = tf.keras.layers.MaxPooling2D((2, 2))(do1)
    c2 = tf.keras.layers.Conv2D(2*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp1)
    do2 = tf.keras.layers.Dropout(0.1)(c2)

    if n_depth>2:
        mp2 = tf.keras.layers.MaxPooling2D((2, 2))(do2)
        c3 = tf.keras.layers.Conv2D(4*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp2)
        do3 = tf.keras.layers.Dropout(0.2)(c3)

        if n_depth>3:
            mp3 = tf.keras.layers.MaxPooling2D((2, 2))(do3)
            c4 = tf.keras.layers.Conv2D(8*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp3)
            do4 = tf.keras.layers.Dropout(0.2)(c4)

            c4 = tf.keras.layers.Conv2D(8*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do4)
            do4 = tf.keras.layers.Dropout(0.2)(c4)

            ct3 = tf.keras.layers.Conv2DTranspose(4*n_channel, (2, 2), strides=(2, 2), padding='same')(do4)
            if 3 in skip:
                ct3 = tf.keras.layers.concatenate([ct3, do3], axis=3)
            c3 = tf.keras.layers.Conv2D(4*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct3)
            do3 = tf.keras.layers.Dropout(0.2)(c3)

        else:
            c3 = tf.keras.layers.Conv2D(4*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do3)
            do3 = tf.keras.layers.Dropout(0.2)(c3)

        ct2 = tf.keras.layers.Conv2DTranspose(2*n_channel, (2, 2), strides=(2, 2), padding='same')(do3)
        if 2 in skip:
            ct2 = tf.keras.layers.concatenate([ct2, do2], axis=3)
        c2 = tf.keras.layers.Conv2D(2*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct2)
        do2 = tf.keras.layers.Dropout(0.1)(c2)

    else:
        c2 = tf.keras.layers.Conv2D(2*n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do2)
        do2 = tf.keras.layers.Dropout(0.1)(c2)

    ct1 = tf.keras.layers.Conv2DTranspose(n_channel, (2, 2), strides=(2, 2), padding='same')(do2)
    if 1 in skip:
        ct1 = tf.keras.layers.concatenate([ct1, do1], axis=3)

    c1 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct1)
    do1 = tf.keras.layers.Dropout(0.1)(c1)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(do1)

    def loss_func(y_true, y_pred):
        if Params_loss is None:
            return total_loss(y_true, y_pred)
        else:
            return total_loss(y_true, y_pred, DL=Params_loss['DL'], BCE=Params_loss['BCE'], \
                FL=Params_loss['FL'], gamma=Params_loss['gamma'], alpha=Params_loss['alpha'])

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_func, metrics=[dice_loss])
    return model


def get_shallow_unet_more_equal(size=None, n_depth=3, n_channel=4, skip=[1], activation='elu', Params_loss=None):
    '''Get a shallow U-Net model. This function has some options to vary the model, 
        including the number of resolution depth ("n_depth"), number of channels per featrue map ("n_channel"), 
        number of skip connections ("skip"), and choice of activation function ("activation").
        The number of channels are the same on each depth. 

    Inputs: 
        size (int, default to None): Lateral size of each dimension of the input layer.
            Usually using None is sufficient, and using None can only cover rectangular input. 
        n_depth (int, default to 3): Number of resolution depth. Can be 2, 3, or 4
        n_channel (int, default to 4): Number of channels per feature map in the first resolution depth.
            For deeper depths, the numbers of feature map will double for every depth. 
        skip (list of int, default to [1]]): Indeces of resolution depths that use skip connections.
            The shallowest depth is 1, and 1 should usually be in "skip".
        activation (str, default to 'elu): activation function. Can be 'elu' or 'relu'.
        Params_loss(dict, default to None): parameters of the loss function "total_loss". Only used in training.
            Params_loss['DL'](float): Coefficient of dice loss in the total loss
            Params_loss['BCE'](float): Coefficient of binary cross entropy in the total loss
            Params_loss['FL'](float): Coefficient of focal loss in the total loss
            Params_loss['gamma'] (float): first parameter of focal loss
            Params_loss['alpha'] (float): second parameter of focal loss

    Outputs:
        model: the CNN model. 
    '''
    if activation=='elu':
        activation=tf.keras.activations.elu
    elif activation=='relu':
        activation=tf.keras.activations.relu
    inputs = tf.keras.layers.Input((size, size, 1))

    c1 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(inputs)
    do1 = tf.keras.layers.Dropout(0.1)(c1)

    mp1 = tf.keras.layers.MaxPooling2D((2, 2))(do1)
    c2 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp1)
    do2 = tf.keras.layers.Dropout(0.1)(c2)

    if n_depth>2:
        mp2 = tf.keras.layers.MaxPooling2D((2, 2))(do2)
        c3 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp2)
        do3 = tf.keras.layers.Dropout(0.2)(c3)

        if n_depth>3:
            mp3 = tf.keras.layers.MaxPooling2D((2, 2))(do3)
            c4 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(mp3)
            do4 = tf.keras.layers.Dropout(0.2)(c4)

            c4 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do4)
            do4 = tf.keras.layers.Dropout(0.2)(c4)

            ct3 = tf.keras.layers.Conv2DTranspose(n_channel, (2, 2), strides=(2, 2), padding='same')(do4)
            if 3 in skip:
                ct3 = tf.keras.layers.concatenate([ct3, do3], axis=3)
            c3 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct3)
            do3 = tf.keras.layers.Dropout(0.2)(c3)

        else:
            c3 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do3)
            do3 = tf.keras.layers.Dropout(0.2)(c3)

        ct2 = tf.keras.layers.Conv2DTranspose(n_channel, (2, 2), strides=(2, 2), padding='same')(do3)
        if 2 in skip:
            ct2 = tf.keras.layers.concatenate([ct2, do2], axis=3)
        c2 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct2)
        do2 = tf.keras.layers.Dropout(0.1)(c2)

    else:
        c2 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(do2)
        do2 = tf.keras.layers.Dropout(0.1)(c2)

    ct1 = tf.keras.layers.Conv2DTranspose(n_channel, (2, 2), strides=(2, 2), padding='same')(do2)
    if 1 in skip:
        ct1 = tf.keras.layers.concatenate([ct1, do1], axis=3)

    c1 = tf.keras.layers.Conv2D(n_channel, (3, 3), activation=activation, kernel_initializer='he_normal', padding='same')(ct1)
    do1 = tf.keras.layers.Dropout(0.1)(c1)

    outputs = tf.keras.layers.Conv2D(1, (1, 1), activation='sigmoid')(do1)

    def loss_func(y_true, y_pred):
        if Params_loss is None:
            return total_loss(y_true, y_pred)
        else:
            return total_loss(y_true, y_pred, DL=Params_loss['DL'], BCE=Params_loss['BCE'], \
                FL=Params_loss['FL'], gamma=Params_loss['gamma'], alpha=Params_loss['alpha'])

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])
    model.compile(optimizer='adam', loss=loss_func, metrics=[dice_loss])
    return model


if __name__ == '__main__':
    # model = get_shallow_unet_more(n_depth=3, n_channel=4)
    model = get_shallow_unet()
    model.summary()