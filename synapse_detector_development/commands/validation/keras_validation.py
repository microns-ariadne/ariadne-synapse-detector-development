#!/usr/bin/env python
"""
Keras Model Validation

Validate a Keras model with custom layers.

Usage:
    keras_validation.py <model-file> <weights-file>

"""
import json
import os
import sys

from docopt import docopt
from keras.models import model_from_json

sys.path.append(os.path.abspath('.'))

try:
    import custom_layers
except ImportError:
    custom_objects = {}


MODULE_DOC = __doc__


def validate(model_path, weights_path):
    """Validate a standard Keras model with no custom layers.

    Parameters
    ----------
    model_path : str
        Path to the model JSON file.
    weights_path : str
        Path to the model weights HDF5 file.

    Returns
    -------
    0 if successful, 1 otherwise.
    """
    global custom_objects

    with open(model_path, 'r') as f:
        model = model_from_json(
            json.dumps(json.load(f)),
            custom_objects=custom_layers.custom_objects)

    model.load_weights(weights_path)

    return 0


def main(docstring=MODULE_DOC):
    args = docopt(docstring, version='0.0.1')
    validate(args['<model-file>'], args['<weights-file>'])

if __name__ == '__main__':
    sys.exit(main())
