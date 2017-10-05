#!/usr/bin/env python
"""
ARIADNE Synapse Detector Development.

Initialize a new synapse detector project. This will not overwrite an existing
project with the same name found at <path>.

Usage:
    synapse-detector-development init [-n | --name STRING] <path>
    synapse-detector-development init [-h | --help]

Arguments:
    path    Path to the directory to instantiate the new project [default: .]

Options:
    -n, --name     Name of the new synapse detector project [default: synapse-detector]
    -h, --help     Display this screen.

"""

from .base import BaseCommand

import os
import shutil
import sys

import docopt

from rh_logger import logger


def run(path, name='synapse-detector'):
    """Create a new synapse detector project with example files.

    Parameters
    ----------
    path : str
        Path to the new synapse detector project.
    name : str, optional
        Name of the new synapse detector project. Default: 'synapse-detector'

    Returns
    -------
    0 if successful else 1.
    """
    try:
        logger.start_process(
            'Create Synapse Detector Project',
            'Begin')
    except:
        pass

    logger.report_event(
        'Initializing new synapse detector {}'.format(name))

    # Create the project directory
    project_path = create_project_directory(path, name)
    sample_files = os.path.join(os.path.basename(__file__), 'samples')

    # Move the sample files if the project directory was created
    if project_path is not None:
        s1 = copy_file(
            project_path,
            os.path.join(sample_files, 'synapse_detector_training.py'))
        s2 = copy_file(
            project_path,
            os.path.join(sample_files, 'custom_layers.py'))
        s3 = copy_file(
            project_path,
            os.path.join(sample_files, 'classifier.yaml'))
        s4 = copy_file(
            project_path,
            os.path.join(sample_files, 'lab-rh-config.yaml'))

    # Determine if the project was successfully initialized
    if project_path is None and not all([s1, s2, s3, s4]):
        logger.report_event('Could not create synapse detector project')
        logger.report_event('Cleaning up and exiting')
        if project_path is not None:
            shutil.rmtree(project_path)
        return 1

    logger.report_event(
        'Successfully created new synapse detector project at {}'
        .format(project_path))

    return 0


def create_project_directory(path, name):
    """Create a new synapse detector project with example files.

    Parameters
    ----------
    path : str
        Path to the new synapse detector project.
    name : str
        Name of the new synapse detector project.

    Returns
    -------
    The path to the project directory if successful else None.
    """
    try:
        logger.start_process(
            'Create Synapse Detector Project',
            'Create Dir')
    except:
        pass

    newpath = os.path.join(path, name.replace(' ', '_'))

    logger.report_event('Creating project directory at {}'.format(newpath))

    try:
        os.mkdir(newpath)
    except OSError as e:
        logger.report_exception(
            exception=e,
            msg='Could not create directory {}'.format(newpath)
        )
        logger.report_event('Please re-run with new project path or name.')
        newpath = None

    return newpath


def copy_file(project_path, filename):
    """Copy the a starter file to the new project directory.

    Parameters
    ----------
    project_path : str
        Path to the new project directory.
    filename : str
        Name of the file to copy to ``project_path``.

    Returns
    -------
    True if successful, False otherwise.
    """
    try:
        logger.start_process('Create Synapse Detector Project', 'Begin')
    except:
        pass

    logger.report_event(
        'Copying {} to {}'.format(filename, project_path))

    # Copy the sample training file to the
    try:
        shutil.copy(
            os.path.join(
                self.sample_file_path,
                filename),
            project_path)
    except IOError as e:
        logger.report_exception(
            exception=e,
            msg='Could not copy {}.'.format(filename))
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
