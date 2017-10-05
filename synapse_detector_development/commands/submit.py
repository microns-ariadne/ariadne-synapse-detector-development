"""
ARIADNE Synapse Detector Development.

Submit a trained synapse classifier for evaluation.

Usage:
    synapse-detector-development submit <model-file> <weights-file> <metadata-file> [<custom-layer-file>]
    synapse-detector-development submit -h | --help

Arguments:
    model-file          Path to the model structure file (model.json).
    weights-file        Path to the model weights file (weights.h5).
    metadata-file       Path to the metadata file (classifier.yaml).
    custom-layer-file   Path to the custom layer code file, optional (custom_layers.py).

Options:
    -h, --help   Show this screen.
"""
from .base import BaseCommand

import getpass
import json
import os
import sys
try:
    # python 3
    from urllib.parse import urlencode
except ImportError:
    # python 2
    from urllib import urlencode

import h5py
from keras.models import model_from_json
import pycurl
import yaml

from rh_logger import logger


def __init__(self, **kws):
    self.model_path = kws['<model-file>'] if '<model-file>' in kws \
                      else os.path.join(os.getcwd(), 'model.json')
    self.weights_path = kws['<weights-file>'] if '<weights-file>' in kws \
                      else os.path.join(os.getcwd(), 'weights.h5')
    self.metadata = kws['<metadata>'] if '<metadata>' in kws \
                      else os.path.join(os.getcwd(), 'classifier.yaml')

    self.password = getpass.getpass('Enter Submission Site Access Key: ')
    if sys.version_info[0] == 2:
        self.password = unicode(self.password)

    self.name = raw_input('Model Name: ')
    self.author = raw_input('Model Author: ')
    self.desc = raw_input('Short Description of the Model: ')

    self.url = 'http://localhost:5000'
    self.auth_url = '/'.join([self.url, 'login'])
    self.upload_url = '/'.join([self.url, 'upload'])


def run(model_path, weights_path, metadata, custom_layers=None):
    """Submit a trained synapse detector for evaluation.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model (e.g., model.json).
    weights_path : str
        Path to the model weights (e.g., weights.h5).
    metadata : str
        Path to the metadata file (e.g., classifier.yaml)
    custom_layers : str, optional
        Path to the custom layers definition script.

    Returns
    -------
    0 if successful else 1.
    """
    try:
        logger.start_process(
            'Upload Synapse Detector',
            'Begin')
    except:
        pass

    # Verify that this is a valid Keras model
    success = verify_files(
        model_path,
        weights_path,
        metadata,
        custom_layers=custom_layers)

    # If model is invalid, exit early
    if not success:
        return 1

    # Attempt to authenticate and upload the model files
    # Exit if authentication fails
    c = pycurl.Curl()
    c.setopt(c.COOKIEFILE, '')
    if authenticate(c):
        success = upload_files(c)
    else:
        c.close()
        return 1
    c.close()

    if success:
        return 0
    else:
        return 1


def verify_files(model_path, weights_path, metadata, custom_layers=None):
    """Verify that the submitted model is a valid Keras model and metadata.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model (e.g., model.json).
    weights_path : str
        Path to the model weights (e.g., weights.h5).
    metadata : str
        Path to the metadata file (e.g., classifier.yaml)
    custom_layers : str, optional
        Path to the custom layers definition script.

    Returns
    -------
    True if successful else False.
    """
    try:
        logger.start_process(
            'Upload Synapse Detector',
            'Begin')
    except:
        pass

    logger.report_event('Verifying upload files')

    # Verify the metadata and model files independently
    s1 = self.verify_metadata()
    s2 = self.verify_keras_files()

    return (s1 and s2)


def verify_metadata(metadata):
    """Verify a submitted model metadata file.

    Parameters
    ----------
    metadata : str
        Path to the metadata file (e.g., classifier.yaml)

    Returns
    -------
    True if successful else False.
    """
    try:
        logger.start_process(
            'Upload Synapse Detector',
            'Begin')
    except:
        pass

    logger.report_event('Verifying metadata file')

    return True


def verify_keras_files(model_path, weights_path, custom_layers=None):
    """Verify that the submitted model is a valid Keras model.

    Parameters
    ----------
    model_path : str
        Path to the saved Keras model (e.g., model.json).
    weights_path : str
        Path to the model weights (e.g., weights.h5).
    metadata : str
        Path to the metadata file (e.g., classifier.yaml)
    custom_layers : str, optional
        Path to the custom layers definition script.

    Returns
    -------
    True if successful else False.
    """
    try:
        logger.start_process(
            'Upload Synapse Detector',
            'Begin')
    except:
        pass

    logger.report_event('Verifying Keras files')

    try:
        with open(self.model_path, 'r') as f:
            if 
            model = model_from_json(json.dumps(json.load(f)))
    except (OSError, IOError, ValueError, TypeError) as e:
        logger.report_exception(
            exception=e,
            msg='Could not load Keras model from file {}'.format(self.model_path))
        return False

    try:
        model.load_weights(self.weights_path)
    except (IOError, OSError) as e:
        logger.report_exception(
            exception=e,
            msg='Could not load model weights from file {}'.format(self.weights_path))
        return False

    return True


def authenticate(self, c):
    """
    """
    post_data = {'access-key': str(self.password)}
    c.setopt(c.URL, self.auth_url)
    c.setopt(c.POST, 1)
    c.setopt(c.POSTFIELDS, urlencode(post_data))
    c.setopt(c.WRITEFUNCTION, lambda x: None)

    try:
        c.perform()
    except pycurl.error as e:
        logger.report_exception(
            exception=e)
            #msg='Could not authenticate.')
        return False

    logger.report_event(c.getinfo(c.RESPONSE_CODE))
    if int(c.getinfo(c.RESPONSE_CODE)) not in [200, 302]:
        return False

    return True


def upload_files(self, c):
    """
    """
    c.setopt(c.URL, self.upload_url)
    c.setopt(c.POST, 1)
    c.setopt(c.WRITEFUNCTION, lambda x: None)

    c.setopt(c.HTTPPOST, [
        ('model-name', 'TestModel'),
        ('model-author', 'Firstname Lastname'),
        ('model-desc', 'A test model'),
        ('model-file', (c.FORM_FILE, self.model_path)),
        ('model-weights', (c.FORM_FILE, self.weights_path)),
        ('model-metadata', (c.FORM_FILE, self.metadata))])

    try:
        c.perform()
    except pycurl.error as e:
        logger.report_exception(
            exception=e,
            msg='Could not complete upload.')
        return False

    if int(c.getinfo(c.RESPONSE_CODE)) != 200:
        return False

    return True


def main(docstring):
    arguments = docopt(docstring, version='0.0.1')

    try:
        logger.start_process(
            'Create Synapse Detector Project',
            'Begin')
    except:
        pass

    path = os.path.abspath(arguments['<path>']) \
        if arguments['<path>'] is not None \
        else os.getcwd()
    name = arguments['-n'] \
        if arguments['-n'] is not None \
        else 'synapse-detector'

    return run(path, name=name)


if __name__ == '__main__':
    sys.exit(main(__doc__))
