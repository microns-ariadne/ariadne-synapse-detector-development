from .base import BaseCommand

import getpass
import json
import os
import shutil
import subprocess
import sys
import tempfile
try:
    # python 3
    from urllib.parse import urlencode
except ImportError:
    # python 2
    from urllib import urlencode

import h5py
import pycurl
import yaml

from rh_logger import logger


class UploadCommand(BaseCommand):
    """
    """

    def __init__(self, **kws):
        self.model_path = kws['<model-file>'] if '<model-file>' in kws \
                          else os.path.join(os.getcwd(), 'model.json')
        self.weights_path = kws['<weights-file>'] if '<weights-file>' in kws \
                          else os.path.join(os.getcwd(), 'weights.h5')
        self.metadata = kws['<metadata>'] if '<metadata>' in kws \
                          else os.path.join(os.getcwd(), 'classifier.yaml')
        self.custom = kws['<custom-layer-file>'] if '<custom-layer-file>' in kws \
                          else None

        self.password = getpass.getpass('Enter Submission Site Access Key: ')
        if sys.version_info[0] == 2:
            self.password = unicode(self.password)

        self.name = raw_input('Model Name: ')
        self.author = raw_input('Model Author: ')
        self.desc = raw_input('Short Description of the Model: ')

        self.url = 'http://138.197.67.50'
        self.auth_url = '/'.join([self.url, 'login'])
        self.upload_url = '/'.join([self.url, 'upload'])

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

        return True

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
            temp = tempfile.mkdtemp()
            shutil.copy(self.model_path, temp)
            shutil.copy(self.weights_path, temp)
            if self.custom is not None:
                shutil.copy(self.custom, temp)
            try:
                script = os.path.join(
                    os.path.dirname(__file__),
                    'validation',
                    'keras_validation.py')
                model = os.path.join(temp, self.model_path)
                weights = os.path.join(temp, self.weights_path)
                subprocess.call([script, model, weights], cwd=temp)
            except subprocess.CalledProcessError:
                print('Ya done goofed')
        except (OSError, IOError, ValueError, TypeError) as e:
            print(e)
            logger.report_exception(
                exception=e,
                msg='Could not load Keras model from file {}'.format(self.model_path))
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

        post_data = [
            ('model-name', 'TestModel'),
            ('model-author', 'Firstname Lastname'),
            ('model-desc', 'A test model'),
            ('model-file', (c.FORM_FILE, self.model_path)),
            ('model-weights', (c.FORM_FILE, self.weights_path)),
            ('model-metadata', (c.FORM_FILE, self.metadata))]

        if self.custom is not None:
            post_data.append(('custom-layer-file', (c.FORM_FILE, self.custom)))

        c.setopt(c.HTTPPOST, post_data)

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
