"""Declare custom layers in this script.

In order to test a Keras model with custom layers, it is necessary to supply
the custom layer code to the evaluation framework. This file demonstrates how
to set up custom layers for submission using the DepthToSpace3D layer from the
refernce U-Net model.
"""
from keras.layers import Layer


###############################################################################
# Write the Custom Layers

class MyLayer(Layer):
    """Example Keras 1.2 custom layer"""
    def __init__(self, output_shape, **kwargs):
        self.output_dim = output_shape
        super(MyLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        # Create a trainable weight variable for this layer.
        self.W = self.add_weight(shape=(input_shape[1], self.output_dim),
                                 initializer='uniform',
                                 trainable=True)
        super(MyLayer, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, x, mask=None):
        return K.dot(x, self.W)

    def get_output_shape_for(self, input_shape):
        return (input_shape[0], self.output_dim)




###############################################################################
# Register the Custom Layers
#
# The `custom_objects` dictionary below is set up to match the Keras API for
# loading custom layers. Layers are added to the dictionary using a string
# version of the class name as the key and the class itself as the value.

custom_objects = {
    'MyLayer': MyLayer
}

# This line allows the `custom_objects` dictionary and and custom layer classes
# to be imported in other scripts with `from custom_layers import *`. Add your
# own custom layers here as you create them.

__all__ = ['custom_objects', 'MyLayer']

###############################################################################
