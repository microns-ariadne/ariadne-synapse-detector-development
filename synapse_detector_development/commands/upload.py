from .base import BaseCommand

import json
import os

import h5py
from keras.models import model_from_json
import pycurl
import yaml

from rh_logger import logger


class UploadCommand(BaseCommand):
    """
    """

    def __init__(self, **kws):
        self.model_path = kws['model-file'] if 'model-file' in kws \
                          else os.path.join(os.getcwd(), 'model.json')
        self.weights_path = kws['weights-file'] if 'weights-file' in kws \
                          else os.path.join(os.getcwd(), 'weights.h5')
        self.model_path = kws['metadata'] if 'metadata' in kws \
                          else os.path.join(os.getcwd(), 'classifier.yaml')

        self.url = ''

    def run(self):
        try:
            logger.start_process(
                'Upload Synapse Detector',
                'Begin')
        except:
            pass

        success = self.verify_files()

        if not success:
            return False

        c = pycurl.Curl()
        c.setopt(c.COOKIEFILE, '')
        if self.authenticate(c):
            self.upload_files(c)

        c.close()

        return True

    def verify_files(self):
        """
        """
        try:
            logger.start_process(
                'Upload Synapse Detector',
                'Begin')
        except:
            pass

        logger.report_event('Verifying upload files')

        s1 = self.verify_metadata()
        s2 = self.verify_keras_files()

        return (s1 and s2)


    def verify_metadata(self):
        """
        """
        try:
            logger.start_process(
                'Upload Synapse Detector',
                'Begin')
        except:
            pass

        logger.report_event('Verifying metadata file')


    def verify_keras_files(self):
        """
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
        post_data = {'access-key': }
        c.setopt(c.URL, )
        c.setopt(c.POST, 1)
        c.setopt(c.POSTFIELDS, )

        try:
            c.perform()
        except pycurl.error as e:
            logger.report_exception(
                exception=e,
                msg='Could not authenticate.')
            return False

        if int(c.getinfo(c.RESPONSE_CODE)) != 200:
            return False

        return True

    def upload_files(self, c):
        """
        """
        files = [
            ('model-file', (c.FORM_FILE, self.model_path, c.FORM_CONTENTTYPE, 'application/json')),
            ('model-weights', (c.FORM_FILE, self.weights_path, c.FORM_CONTENTTYPE, 'text/plain')),
            ('metadata', (c.FORM_FILE, self.metadata, c.FORM_CONTENTTYPE, 'text/plain'))
        ]

        c.setopt(c.URL, )
        c.setopt(c.POST, 1)
        for f in files:
            c.setopt(c.HTTPPOST, [f])
        c.setopt(pycurl.HTTPHEADER, ['Accept-Language: en'])

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
