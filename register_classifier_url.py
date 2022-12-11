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
            def is_within_directory(directory, target):
                
                abs_directory = os.path.abspath(directory)
                abs_target = os.path.abspath(target)
            
                prefix = os.path.commonprefix([abs_directory, abs_target])
                
                return prefix == abs_directory
            
            def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
            
                for member in tar.getmembers():
                    member_path = os.path.join(path, member.name)
                    if not is_within_directory(path, member_path):
                        raise Exception("Attempted Path Traversal in Tar File")
            
                tar.extractall(path, members, numeric_owner=numeric_owner) 
                
            
            safe_extract(tar, path=temp)
            if len(glob.glob(os.path.join(temp, '*'))) != 3:
                pass
        shutil.rmtree(temp)

    try:
        with open(args.yaml_file, 'r') as f:
            data = yaml.load(f)
    except IOError:
        data = None

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
        yaml.dump(data, f, default_flow_style=False)


if __name__ == '__main__':
    main()
