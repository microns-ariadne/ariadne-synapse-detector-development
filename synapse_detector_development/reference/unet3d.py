from keras.layers import Input, Conv3D, MaxPooling3D, Cropping3D, \
                         UpSampling3D, concatenate, ZeroPadding3D
from keras.models import Model


def unet3d(input_shape=(97, 1496, 1496, 1)):
    i = Input(shape=input_shape)

    # Convolutional downsample unit 1
    conv1 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(i)
    conv2 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(conv2)

    # Convolutional downsample unit 1
    conv3 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(pool1)
    conv4 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv3)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(conv4)

    # Convolutional downsample unit 1
    conv5 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(pool2)
    conv6 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv5)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(conv6)

    # Convolutional upsample unit 1
    conv7 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(pool3)
    conv8 = Conv3D(128, (3, 3, 3), activation='relu', padding='same')(conv7)
    upsample1 = UpSampling3D(size=(1, 2, 2))(conv8)
    conv9 = Conv3D(64, (1, 2, 2), activation='relu', padding='same')(upsample1)

    # Convolutional upsample unit 1
    cs = int((conv6._keras_shape[3] - conv9._keras_shape[3]) / 2)
    csZ = int((conv6._keras_shape[2] - conv9._keras_shape[2]) / 2)
    crop3 = Cropping3D(cropping=((csZ, csZ), (cs, cs), (cs, cs)))(conv6)
    merge1 = concatenate([conv9, crop3], axis=4)
    conv10 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(merge1)
    conv11 = Conv3D(64, (3, 3, 3), activation='relu', padding='same')(conv10)
    upsample2 = UpSampling3D(size=(1, 2, 2))(conv11)
    conv12 = Conv3D(32, (1, 2, 2), activation='relu', padding='same')(upsample2)

    # Convolutional upsample unit 1
    cs = int((conv4._keras_shape[3] - conv12._keras_shape[3]) / 2)
    csZ = int((conv4._keras_shape[2] - conv12._keras_shape[2]) / 2)
    crop2 = Cropping3D(cropping=((csZ, csZ), (cs, cs), (cs, cs)))(conv4)
    merge2 = concatenate([conv12, crop2], axis=4)
    conv13 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(merge2)
    conv14 = Conv3D(32, (3, 3, 3), activation='relu', padding='same')(conv13)
    upsample3 = UpSampling3D(size=(1, 2, 2))(conv14)
    conv15 = Conv3D(16, (1, 2, 2), activation='relu', padding='same')(upsample3)

    # Convolutional upsample unit 1
    cs = int((conv2._keras_shape[3] - conv15._keras_shape[3]) / 2)
    csZ = int((conv2._keras_shape[2] - conv15._keras_shape[2]) / 2)
    crop1 = Cropping3D(cropping=((csZ, csZ), (cs, cs), (cs, cs)))(conv2)
    merge3 = concatenate([conv15, crop1], axis=4)
    conv16 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(merge3)
    conv17 = Conv3D(16, (3, 3, 3), activation='relu', padding='same')(conv16)
    conv18 = Conv3D(1, (1, 1, 1), activation='linear', padding='same')(conv17)

    model = Model(inputs=i, outputs=conv18)

    return model
