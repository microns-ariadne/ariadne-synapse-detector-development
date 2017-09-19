"""Compress a synapse classifier model and its metadata.

This script compresses a pickled synapse classifier, its model file, its,
weights, and a classifier.yaml metadata file, storing them in a gzipped tar
archive. The MD5 hash of the resulting archive is reported.

Usage:
python compress_synapse_classifier.py <classifier> <classifier.yaml> <output filename>
"""
import argparse
import errno
import hashlib
try:
    # Import pickle in Python 2.X
    import cPickle as pickle
except ImportError:
    # Import pickle in Python 3.X
    import pickle
import os
import tarfile
import warnings


from ariadne_microns_pipeline.classifiers.keras_classifier \
     import NormalizeMethod


try:
    FileNotFoundError
except NameError:
    # Define FileNotFoundError for Python2
    FileNotFoundError = OSError


MISMATCH_MESSAGE = """
classifier.yaml field {} and pickled classifier attribute {} do not match.
"""


def parse_args(args=None):
    p = argparse.ArgumentParser()

    p.add_argument('classifier',
                   help='path to the pickled classifier',
                   type=str)

    p.add_argument('output',
                   help='the name of the output tar.gz file',
                   type=str)

    return p.parse_args(args)


def main():
    args = parse_args()

    print("Compressing the following files:")

    if os.path.isfile(args.classifier):
        with open(args.classifier, 'r') as f:
            print('\t- {}'.format(args.classifier))
            classifier = pickle.load(f)
    else:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            args.classifier)

    model_path = classifier.model_path
    weights_path = classifier.weights_path

    if os.path.isfile(model_path):
        print('\t- {}'.format(model_path))
    else:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            model_path)

    if os.path.isfile(weights_path):
        print('\t- {}'.format(weights_path))
    else:
        raise FileNotFoundError(
            errno.ENOENT,
            os.strerror(errno.ENOENT),
            weights_path)

    print('Compressing files')
    with tarfile.open(args.output, 'w') as tar:
        tar.add(args.classifier, arcname=os.path.basename(args.classifier))
        tar.add(model_file, arcname=os.path.basename(model_file))
        tar.add(weights_file, arcname=os.path.basename(weights_file))

    print('Done')
    print('The compressed archive has been saved to: {}'.format(args.output))
    print('MD5 Hash: {}'.format(hashlib.md5(local_path).hexdigest()))

if __name__ == '__main__':
    main()
