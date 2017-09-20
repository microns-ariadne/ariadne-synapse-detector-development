"""Reference U-Net implementation.
"""
from __future__ import absolute_import

from keras.layers import Input, merge, Convolution3D, MaxPooling3D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import ELU
from keras.models import Model
from keras.optimizers import SGD, Adam
from keras.callbacks import Callback
import keras.backend as K
from keras.engine import Layer, InputSpec
import numpy as np

def unet():
    """Create a reference 3D U-Net model.

    Returns
    -------
    unet : `keras.model.Model`
    """
    if K._BACKEND == 'tensorflow':
        INPUT_SHAPE = (17, 512, 512, 1)
        OUTPUT_SHAPE = (17, 512, 512, 1)
    else:
        INPUT_SHAPE = (1, 11, 512, 512)
        OUTPUT_SHAPE = (1, 11, 512, 512)

    x = Input(shape=INPUT_SHAPE)
    first = Convolution3D(10, 3, 3, 3, border_mode='same', bias=True)(x)
    middle = unet_layers(first, 10, 10, depth=3, feature_map_mul=3)
    out = Convolution3D(1, 1, 1, 1, activation='sigmoid')(middle)
    model = Model(input=x, output=out)
    return model


def unet_layers(input, num_features, num_input_features,
         depth=3, feature_map_mul=3):
    # bring input up to internal number of features
    increase_features = Convolution3D(num_features, 1, 1, 1)(input)

    # preprocessing block
    chain1 = residual_block(increase_features, num_features)
    if depth == 0:
        return chain1

    # recurse to next terrace
    downsampled = MaxPooling3D(pool_size=(1, 2, 2))(chain1)
    nested = unet(downsampled, feature_map_mul * num_features, num_features,
                  depth=(depth - 1), feature_map_mul=feature_map_mul)
    # bring up to 4x features
    post_nested = Convolution3D(4 * num_features, 1, 1, 1)(nested)
    upsampled = DepthToSpace3D()(post_nested)

    # merge preprocessing block and nested block
    merged = merge([chain1, upsampled], mode='sum')

    # postprocessing block
    chain2 = residual_block(merged, num_features)

    # take back down to input size
    decrease_features = Convolution3D(num_input_features, 1, 1, 1)(chain2)

    # merge
    return merge([input, decrease_features], mode='sum')


def residual_block(input, num_feature_maps, filter_size=3):
    conv_1 = BatchNormalization(axis=1, mode=2)(input)
    conv_1 = ELU()(conv_1)
    conv_1 = Convolution3D(num_feature_maps, filter_size, filter_size, filter_size,
                           border_mode='same', bias=False)(conv_1)

    conv_2 = BatchNormalization(axis=1, mode=2)(conv_1)
    conv_2 = ELU()(conv_2)
    conv_2 = Convolution3D(num_feature_maps, filter_size, filter_size, filter_size,
                           border_mode='same', bias=True)(conv_2)

    return merge([input, conv_2], mode='sum')


def weighted_mse(y_true, y_pred):
    # per batch positive fraction, negative fraction (0.5 = ignore)
    pos_mask = K.cast(y_true > 0.75, 'float32')
    neg_mask = K.cast(y_true < 0.25, 'float32')
    num_pixels = K.cast(K.prod(K.shape(y_true)[1:]), 'float32')
    pos_fracs = K.clip((K.sum(pos_mask, axis=[1, 2, 3, 4], keepdims=True) /
                        num_pixels),
                       0.01, 0.99)
    neg_fracs = K.clip((K.sum(neg_mask, axis=[1, 2, 3, 4], keepdims=True) /
                        num_pixels),
                       0.01, 0.99)

    pos_fracs = maybe_print(pos_fracs, "positive fraction")

    # chosen to sum to 1 when multiplied by their fractions, assuming no ignore
    pos_weight = maybe_print(1.0 / (2 * pos_fracs), "positive weight")
    neg_weight = maybe_print(1.0 / (2 * neg_fracs), "negative weight")

    per_pixel_weights = pos_weight * pos_mask + neg_weight * neg_mask
    per_pixel_weighted_sq_error = K.square(y_true - y_pred) * per_pixel_weights

    batch_weighted_mse = K.mean(per_pixel_weighted_sq_error,
                                axis=[1, 2, 3, 4])

    return K.mean(batch_weighted_mse)


def maybe_print(tensor, msg, do_print=False):
    if do_print:
        return K.print_tensor(tensor, msg)
    else:
        return tensor


class DepthToSpace3D(Layer):
    '''Cropping layer for 3D input (e.g. multichannel picture).
    '''
    input_ndim = 5

    def __init__(self, block_size=2, dim_ordering=K.image_dim_ordering(), **kwargs):
        super(DepthToSpace3D, self).__init__(**kwargs)
        assert dim_ordering in {'tf', 'th'}, 'dim_ordering must be in {tf, th}'
        self.block_size = block_size
        self.dim_ordering = dim_ordering
        self.input_spec = [InputSpec(ndim=5)]

    def get_output_shape_for(self, input_shape):
        block_size_sq = self.block_size ** 2
        if self.dim_ordering == 'tf':
            assert K._BACKEND == 'tensorflow'
            assert input_shape[4] % block_size_sq == 0
            return (input_shape[0],
                    input_shape[1],
                    input_shape[2] * self.block_size,
                    input_shape[3] * self.block_size,
                    input_shape[4] // block_size_sq)
        elif self.dim_ordering == 'th':
            assert K._BACKEND == 'theano'
            assert input_shape[1] % block_size_sq == 0
            return (input_shape[0],
                    input_shape[1] // block_size_sq,
                    input_shape[2],
                    input_shape[3] * self.block_size,
                    input_shape[4] * self.block_size)
        else:
            raise Exception('Invalid dim_ordering: ' + self.dim_ordering)

    def call(self, x, mask=None):
        if self.dim_ordering == 'tf':
            Xsplit = K.tf.unpack(x, axis=1)
            return K.tf.pack([K.tf.depth_to_space(subx, self.block_size) for subx in Xsplit], axis=1)
        else:
            block_size = self.block_size
            b, k, d, r, c = x.shape
            # x.shape is a theano expression
            out = K.reshape(K.zeros_like(x),
                            (b, k // (block_size ** 2), d, r * block_size, c * block_size))
            for i in range(block_size):
                for j in range(block_size):
                    out = K.T.set_subtensor(out[:, :, :, i::block_size, j::block_size],
                                            x[:, (block_size * i + j)::(block_size ** 2), :, :, :])
            return out

    def get_config(self):
        config = {'block_size': self.block_size}
        base_config = super(DepthToSpace3D, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
