from __future__ import print_function

from .base import BaseCommand

import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import sys

import yaml

from ariadne_microns_pipeline.classifiers.keras_classifier \
    import KerasClassifier, NormalizeMethod
from rh_logger import logger


class PickleCommand(BaseCommand):
    def __init__(self, **kws):
        self.model_path = os.path.abspath(kws['<model-path>']) \
            if kws['<model-file>'] is not None else None
        self.weights_path = os.path.abspath(kws['<weights-file>']) \
            if kws['<weights-file>'] is not None else None
        self.metadata = os.path.abspath(kws['<metadata>']) \
            if kws['<metadata>'] is not None else None

    def run(self):
        try:
            logger.start_process(
                'Pickle Synapse Detector',
                'Begin')
        except:
            pass

        try:
            logger.report_event('Loading {}'.format(self.metadata))
            with open(self.metadata, 'r') as f:
                params = yaml.load(f)["classifier"]

            transpose = map(
                lambda _: None if isinstance(_, basestring) and _.lower() == "none" else _,
                params["transpose"])

            logger.report_event('Creating KerasClassifier')
            k = KerasClassifier(
                    model_path=params["model-path"],
                    weights_path=params["weights-path"],
                    xypad_size=params["xy-pad-size"],
                    zpad_size=params["z-pad-size"],
                    block_size=params["block-size"],
                    normalize_method=NormalizeMethod[params["normalize-method"]],
                    downsample_factor=params["downsample-factor"],
                    xy_trim_size=params["xy-trim-size"],
                    z_trim_size=params["z-trim-size"],
                    classes=params["classes"],
                    stretch_output=params["stretch-output"],
                    invert=params["invert"],
                    split_positive_negative=params["split-positive-negative"],
                    normalize_offset=params["normalize-offset"],
                    normalize_saturation_level=params["normalize-saturation-level"],
                    transpose=transpose
                )

        except (OSError, IOError, TypeError, ValueError, KeyError) as e:
            print('{}'.format(e), file=sys.stderr)
            return (False, 'Could not load pipeline KerasClassifier from metadata')

        logger.report_event(
            'Saving classifier to {}'.format(os.path.abspath('synapse-classifier.pkl')))
        try:
            with open('synapse-classifier.pkl', 'w') as f:
                pickle.dump(k, f)
        except pickle.PicklingError as e:
            print('Unable to pickle the classifier')
