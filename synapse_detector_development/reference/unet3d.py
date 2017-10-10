from keras.layers import Input, Convolution3D, MaxPooling3D, Cropping3D, \
                         UpSampling3D, merge, ZeroPadding3D
from keras.models import Model


def unet3d(input_shape=(97, 1496, 1496, 1)):
    i = Input(shape=input_shape)

    # Convolutional downsample unit 1
    conv1 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(i)
    conv2 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv1)
    pool1 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(conv2)

    # Convolutional downsample unit 1
    conv3 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(pool1)
    conv4 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv3)
    pool2 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(conv4)

    # Convolutional downsample unit 1
    conv5 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(pool2)
    conv6 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv5)
    pool3 = MaxPooling3D(pool_size=(1, 2, 2), strides=(1, 2, 2))(conv6)

    # Convolutional upsample unit 1
    conv7 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(pool3)
    conv8 = Convolution3D(128, 3, 3, 3, activation='relu', border_mode='same')(conv7)
    upsample1 = UpSampling3D(size=(1, 2, 2))(conv8)
    conv9 = Convolution3D(64, 1, 2, 2, activation='relu', border_mode='same')(upsample1)

    # Convolutional upsample unit 1
    cs = int((conv6._keras_shape[3] - conv9._keras_shape[3]) / 2)
    csZ = int((conv6._keras_shape[2] - conv9._keras_shape[2]) / 2)
    crop3 = Cropping3D(cropping=((csZ, csZ), (cs, cs + 1), (cs, cs + 1)))(conv6)
    merge1 = merge([conv9, crop3], mode='concat', concat_axis=1)
    conv10 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(merge1)
    conv11 = Convolution3D(64, 3, 3, 3, activation='relu', border_mode='same')(conv10)
    upsample2 = UpSampling3D(size=(1, 2, 2))(conv11)
    conv12 = Convolution3D(32, 1, 2, 2, activation='relu', border_mode='same')(upsample2)

    # Convolutional upsample unit 1
    cs = int((conv4._keras_shape[3] - conv12._keras_shape[3]) / 2)
    csZ = int((conv4._keras_shape[2] - conv12._keras_shape[2]) / 2)
    crop2 = Cropping3D(cropping=((csZ, csZ), (cs, cs), (cs, cs)))(conv4)
    merge2 = merge([conv12, crop2], mode='concat', concat_axis=1)
    conv13 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(merge2)
    conv14 = Convolution3D(32, 3, 3, 3, activation='relu', border_mode='same')(conv13)
    upsample3 = UpSampling3D(size=(1, 2, 2))(conv14)
    conv15 = Convolution3D(16, 1, 2, 2, activation='relu', border_mode='same')(upsample3)

    # Convolutional upsample unit 1
    cs = int((conv2._keras_shape[3] - conv15._keras_shape[3]) / 2)
    csZ = int((conv2._keras_shape[2] - conv15._keras_shape[2]) / 2)
    crop1 = Cropping3D(cropping=((csZ, csZ), (cs, cs), (cs, cs)))(conv2)
    merge3 = merge([conv15, crop1], mode='concat', concat_axis=1)
    conv16 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(merge3)
    conv17 = Convolution3D(16, 3, 3, 3, activation='relu', border_mode='same')(conv16)
    conv18 = Convolution3D(1, 1, 1, 1, activation='linear', border_mode='same')(conv17)

    model = Model(input=[i], output=[conv18])

    return model
