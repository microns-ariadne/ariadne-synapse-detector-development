from .base import BaseCommand

import os
import shutil
import sys

from rh_logger import logger


class CreateCommand(BaseCommand):
    """Create a new training file for synapse detector development.

    Parameters
    ----------
    name : str, optional
        Name of the new synapse detector
    path : str, optional
        Path to initialize the new synapse detector.

    Attributes
    ----------
    name : str
        Name of the new synapse detector
    path : str
        Path to initialize the new synapse detector.
    sample_file_path : str
        Path to sample files copied to the created project directory.
    """

    def __init__(self, **kws):
        self.path = os.path.abspath(kws['<path>']) if '<path>' in kws and kws['<path>'] is not None \
                    else os.getcwd()

        self.name = kws['<name>'] if '<name>' in kws else 'synapse-detector'

        self.meta_only = kws['--metadata-only'] if '--metadata-only' in kws \
                         else False

        self.sample_file_path = os.path.join(
            os.path.dirname(__file__),
            'samples')

    def run(self):
        """
        """
        try:
            logger.start_process(
                'Create Synapse Detector Project',
                'Begin')
        except:
            pass

        logger.report_event(
            'Initializing new synapse detector {}'.format(self.name))

        if self.meta_only:
            self.copy_metadata_file(os.getcwd())
            logger.report_event(
                'Created metadata file at {}'.format(os.getcwd()))
            return True

        project_path = self.create_project_directory()

        if project_path is not None:
            s1 = self.copy_sample_file(project_path)
            s2 = self.copy_metadata_file(project_path)
            s3 = self.copy_custom_layer_file(project_path)
            s4 = self.copy_rhconfig(project_path)

        if project_path is None and not all([s1, s2, s3, s4]):
            logger.report_event('Could not create synapse detector project')
            logger.report_event('Cleaning up and exiting')
            if project_path is not None:
                shutil.rmtree(project_path)
            sys.exit(1)

        logger.report_event(
            'Successfully created new synapse detector project at {}'
            .format(project_path))

    def create_project_directory(self):
        try:
            logger.start_process(
                'Create Synapse Detector Project',
                'Create Dir')
        except:
            pass

        newpath = os.path.join(self.path, self.name.replace(' ', '_'))

        logger.report_event('Creating project directory at {}'.format(newpath))

        try:
            os.mkdir(newpath)
        except OSError as e:
            logger.report_exception(
                exception=e,
                msg='Could not create directory {}'.format(self.path)
            )
            logger.report_event('Please re-run with new project path or name.')
            sys.exit(1)

        return newpath

    def copy_sample_file(self, project_path):
        try:
            logger.start_process('Create Synapse Detector Project', 'Begin')
        except:
            pass

        logger.report_event(
            'Copying sample training file to {}'.format(project_path))

        # Copy the sample training file to the
        try:
            shutil.copy(
                os.path.join(
                    self.sample_file_path,
                    'synapse_detector_training.py'),
                project_path)
        except IOError as e:
            logger.report_exception(
                exception=e,
                msg='Could not copy sample training file.')
            return False

        logger.report_event(
            'Sample File: {}'.format(
                os.path.join(project_path, 'synapse_detector_training.py')))

        return True

    def copy_custom_layer_file(self, project_path):
        try:
            logger.start_process('Create Synapse Detector Project', 'Begin')
        except:
            pass

        logger.report_event(
            'Copying sample training file to {}'.format(project_path))

        # Copy the sample training file to the
        try:
            shutil.copy(
                os.path.join(
                    self.sample_file_path,
                    'custom_layers.py'),
                project_path)
        except IOError as e:
            logger.report_exception(
                exception=e,
                msg='Could not copy sample training file.')
            return False

        logger.report_event(
            'Sample File: {}'.format(
                os.path.join(project_path, 'synapse_detector_training.py')))

        return True

    def copy_metadata_file(self, project_path):
        try:
            logger.start_process('Create Synapse Detector Project', 'Begin')
        except:
            pass

        logger.report_event(
            'Copying sample metadata file to {}'.format(project_path))

        try:
            shutil.copy(
                os.path.join(self.sample_file_path, 'example_classifier.yaml'),
                os.path.join(project_path, 'classifier.yaml'))
        except IOError as e:
            logger.report_exception(
                exception=e)
            return False

        logger.report_event(
            'Metadata File: {}'.format(
                os.path.join(project_path, 'classifier.yaml')))

        return True

    def copy_rhconfig(self, project_path):
        try:
            logger.start_process('Create Synapse Detector Project', 'Begin')
        except:
            pass

        logger.report_event(
            'Copying sample metadata file to {}'.format(project_path))

        try:
            shutil.copy(
                os.path.join(self.sample_file_path, 'lab-rh-config.yaml'),
                project_path)
        except IOError as e:
            logger.report_exception(
                exception=e)
            return False

        logger.report_event(
            'rh-config File: {}'.format(
                os.path.join(project_path, 'lab-rh-config.yaml')))

        return True

