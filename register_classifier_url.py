import argparse
import glob
import hashlib
import os
import shutil
import sys
import tarfile
import tempfile
import time
import warnings

import yaml


def parse_args(args=None):
    p = argparse.ArgumentParser()

    p.add_argument('url',
                   help='URL to download the model from',
                   type=str)

    p.add_argument('classifier',
                   help='Path to the compressed classifier',
                   type=str)

    p.add_argument('author',
                   help='Name of the model author',
                   type=str)

    p.add_argument('yaml_file',
                   help='Path to the YAML file that will store this metadata',
                   type=str)

    return p.parse_args(args)


def main():
    args = parse_args()

    if args.yaml_file is None:
        args.yaml_file = os.path.join(
            os.path.dirname(__file__),
            'urls',
            os.path.splitext(os.path.basename(args.classifier))[0])

    if os.path.isfile(args.yaml_file):
        warnings.warn()

    if not tarfile.is_tarfile(args.classifier):
        print("Error: {} is not a compressed classifier".format(args.classifier))
    else:
        temp = tempfile.mkdtemp()
        with tarfile.open(args.classifier, 'r') as tar:
            tar.extractall(path=temp)
            if len(glob.glob(os.path.join(temp, '*'))) != 3:
                pass
        shutil.rmtree(temp)

    with open(args.yaml_file, 'r') as f:
        data = yaml.load(f)

    if data is None:
        data = {'models': []}

    model = {}
    model['url'] = args.url
    with open(args.classifier, 'r') as f:
        model['hash'] = hashlib.md5(f.read()).hexdigest()
    model['author'] = args.author
    model['submission-date'] = time.time()

    data['models'].append(model)
    with open(args.yaml_file, 'w') as f:
        yaml.dump(data, f)


if __name__ == '__main__':
    main()
