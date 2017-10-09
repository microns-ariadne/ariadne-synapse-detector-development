from .base import BaseCommand

import os
import shutil
import subprocess
import sys
import tempfile

from rh_logger import logger


class EvaluateCommand(BaseCommand):
    def __init__(self, **kws):
        self.model_path = os.path.abspath(kws['<model-file>']) \
            if '<model-file>' in kws and kws['<model-file>'] is not None \
            else None

        self.weights_path = os.path.abspath(kws['<weights-file>']) \
            if '<weights-file>' in kws and kws['<weights-file>'] is not None \
            else None

        self.metadata = os.path.abspath(kws['<metadata>']) \
            if '<metadata>' in kws and kws['<metadata>'] is not None \
            else None

        self.custom_layers = os.path.abspath(kws['<custom-layer-file>']) \
            if '<custom-layer-file>' in kws and kws['<custom-layer-file>'] is not None \
            else None

        self.rh_config = os.path.abspath(kws['<rh-config>']) \
            if '<rh-config>' in kws and kws['<rh-config>'] is not None \
            else os.path.join(os.path.dirname(__file__), samples, 'lab-rh-config.yaml')

    def run(self):
        temp = self.prep()
        self.pickle_classifier(temp)
        self.evaluate(temp)
        self.cleanup(temp)

    def prep(self):
        temp = tempfile.mkdtemp()
        shutil.copy(self.model_path, temp)
        shutil.copy(self.weights_path, temp)
        shutil.copy(self.metadata, temp)
        shutil.copy(self.custom_layers, temp)
        shutil.copy(self.rh_config, temp)
        return temp

    def pickle_classifier(self, temp):
        try:
            meta = os.path.join(temp, os.path.basename(self.metadata))
            args = ['synapse-detector-development', 'pickle', meta]
            subprocess.call(args, cwd=temp)
        except subprocess.CalledProcessError:
            print('fuck fuck fuck')

    def evaluate(self, temp):
        script = os.path.join(
            os.path.dirname(__file__),
            'samples',
            'run_synapse.sh')

        env = {
            'MICRONS_TMP_DIR': temp,
            'MICRONS_ROOT_DIR': temp,
            'RH_CONFIG_FILENAME': os.path.join(
                temp,
                os.path.basename(self.rh_config))
        }

        for k in env:
            os.putenv(k, env[k])

        args = ['bash', script]

        try:
            subprocess.check_call(args, cwd=temp, env=dict(os.environ, **env))
        except subprocess.CalledProcessError:
            print('AAAAAAAHHHHH IT FAILED!!!!')

    def compute_statistics(self):
        pass

    def cleanup(self, temp):
        shutil.rmtree(temp)
