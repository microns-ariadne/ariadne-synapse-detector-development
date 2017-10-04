"""This script is an example script for training synapse detectors.

The examples in this script will show how to train a Keras classifier for
synapse detection and save the trained model for submission to the evaluation
framework at the University of Notre Dame.
"""
import json

###############################################################################
# Setup
#
# These are utility functions for loading the training data and training the
# model.
#
# load(): loads a dataset from a set of images, an HDF5 file, or a JSON path
#         specification
#
# augmenting_generator(): create augmented training samples (randomly scaled,
#         rotated, flipped, transposed, etc.)
#
# train_model(): train your model and save the results for submission
#
# unet(): A reference 3D U-Net implementation
# weighted_mse(): a custom loss function to use for training

from synapse_detector_development.dataset import load
from synapse_detector_development.generators import augmenting_generator
from synapse_detector_development.model import train_model
from synapse_detector_development.reference import unet3d

# This imports the custom layers defined in `custom_layers.py`.
# Be sure to define all custom layers in `custom_layers.py` for submission
# otherwise the pipeline will not be able to load your model.
from custom_layers import *

###############################################################################




###############################################################################
# Put other imports here.
from keras.optimizers import Adam

###############################################################################




###############################################################################
# Model Definition
#
# Define your model in this function. Right now, the function loads in a
# reference U-Net implementation, however any model may be created here
# using the standard Keras layers.

def create_model(in_shape, out_shape):
    model = unet3d()
    model.compile(loss='cosine_proximity', optimizer='adam')
    return model


###############################################################################




###############################################################################
# Training and Saving the Model
#
# In the main() function, the model is trained on augmented synapse data, and
# its architecture and weights are saved in a format that the evaluation
# framework understands.

def main():
    # Define the input and output shapes of your model.
    INPUT_SHAPE = (17, 512, 512, 1)
    OUTPUT_SHAPE = (17, 512, 512, 1)

    model = create_model(INPUT_SHAPE, OUTPUT_SHAPE)
    raw, gt, dist = load()
    # The load function can also be provided paths manually, like so:
    # load(dataset=<path-to-raw-data>, gt=<path-to-gt)

    # Create the training data generator
    train_data = augmenting_generator(raw, gt, dist, INPUT_SHAPE,
                                      OUTPUT_SHAPE, 50)

    # Train the model with 50 batches per eposh over 50 epochs
    model.fit_generator(train_data, 50, 50, verbose=2)

    # Save the model to file. Rename the fiel as you see fit.
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights()


if __name__ == '__main__':
    main()