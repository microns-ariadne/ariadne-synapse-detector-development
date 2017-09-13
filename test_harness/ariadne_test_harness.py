import argparse
import csv
# from email.mime.text import MIMEText
import glob
import hashlib
import json
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import re
# import smtplib
import subprocess
import sys
import tarfile
import tempfile
import urllib

import h5py
import numpy as np
import yaml

from ariadne_microns_pipeline.targets.volume_target import DestVolumeReader
import rh_logger


def parse_args(args=None):
    p = argparse.ArgumentParser()

    p.add_argument('-d', '--database',
                   help='path to the JSON database file',
                   type=str)

    p.add_argument('-f', '--url-files',
                   help='path to model url file or directory',
                   type=str)

    p.add_argument('--aggregate-classifier',
                   help='path to the aggregate classifier',
                   type=str)

    p.add_argument('--membrane-classifier',
                   help='path to the membrane classifier',
                   type=str)

    p.add_argument('--membrane-classifier-model',
                   help='path to the membrane classifier model file',
                   type=str)

    p.add_argument('--membrane-classifier-weights',
                   help='path to the membrane classifier weights file',
                   type=str)

    return p.parse_args(args)


def prep(agg, membrane, model, weights):
    """Prepare a workspace for the test harness.

    Creates temporary directories and copies non-synapse model files down. Also
    sets the paths of the non-synapse model files to match the copied temp
    dir locations.

    Parameters
    ----------
    agg : str
        Path to the pickled aggregate classifier.
    membrane : str
        Path to the pickled membrane classifier.
    model : str
        Path to the membrane classifier model JSON file.
    weights: str
        Pth to the membrane classifier weghts HDF5 file.

    Returns
    -------
    tempdir : str
        Path to the temporary directory.

    Notes
    -----
    If at any point during the prep an error is encountered, this program will
    exit. Errors at this stage indicate a more endemic issue that will affect
    testing all models.
    """
    rh_logger.logger.report_event("Preparing workspace")

    # Set up a temporary workspace
    tempdir = tempfile.mkdtemp(suffix='_ariadne_test_harness')
    rh_logger.logger.report_event(
        'Setting up temporary directories in {}'.format(tempdir))
    temp_classifiers = os.path.join(tempdir, 'classifiers')

    try:
        os.mkdir(temp_classifiers)
        os.mkdir(os.path.join(tempdir, 'results'))
        os.mkdir(os.path.join(tempdir, 'test_subject'))
    except OSError as e:
        rh_logger.logger.report_exception(exception=e,
                                          msg="Could not create directories")
        sys.exit(1)

    # Copy the aggregate classifier, membrane classifier, membrane classifier
    # model file, and membrane classifier weights file to the temp workspace
    try:
        rh_logger.logger.report_event('Copying aggregate classifier')
        shutil.copy(
            os.path.join(classifier, '2017-05-04_membrane_2017-05-04_synapse.pkl'),
            os.path.join(temp_classifiers,
                         '2017-05-04_membrane_2017-05-04_synapse.pkl')
        )

        rh_logger.logger.report_event('Copying membrane classifier')
        shutil.copy(
            os.path.join(classifier, '2017-05-04_membrane.pkl'),
            os.path.join(temp_classifiers, '2017-05-04_membrane.pkl')
        )

        rh_logger.logger.report_event('Copying membrane classifier model file')
        shutil.copy(
            os.path.join(classifier,
                         '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'),
            os.path.join(temp_classifiers,
                         '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json')
        )

        rh_logger.logger.report_event('Copying membrane classifier weights file')
        shutil.copy(
            os.path.join(classifier, '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'),
            os.path.join(temp_classifiers, '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5')
        )

    except IOError as e:
        rh_logger.logger.report_exception(exception=e,
                                          msg="Could not copy classifier files")
        sys.exit(1)

    # Set the filepaths in the aggregate and membrane classifiers to point to
    # the temp workspace
    try:
        with open(os.path.join(temp_classifiers, '2017-05-04_membrane_2017-05-04_synapse.pkl'), 'r') as f:
            c = pickle.load(f)

        c.pickle_paths[0] = os.path.join(temp_classifiers, '2017-05-04_membrane.pkl')
        with open(os.path.join(temp_classifiers, '2017-05-04_membrane_2017-05-04_synapse.pkl'), 'w') as f:
            pickle.dump(c, f)

        with open(os.path.join(temp_classifiers, '2017-05-04_membrane.pkl'), 'r') as f:
            c = pickle.load(f)

        c.model_path = os.path.join(temp_classifiers, '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json')
        c.weights_path = os.path.join(temp_classifiers, '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5')
        with open(os.path.join(temp_classifiers, '2017-05-04_membrane.pkl'), 'w') as f:
            pickle.dump(c, f)
    except TypeError as e:
        rh_logger.logger.report_exception(exception=e,
                                          msg='File could not be pickled.')
        sys.exit(1)

    return tempdir


def load_urls(urlpath):
    """Load model URL/hash pairs from file.

    Parameters
    ----------
    urlpath : str
        Path to the YAML file to load urls from or to a directory containing
        multiple such files.

    Returns
    -------
    pairs : list of tuple
        A list of 2-tuples with the URL as the first element and the model
        archive hash as the second.
    """
    rh_logger.report_event("Loading URL/hash pairs from {}".format(urlpath))
    pairs = []
    files = []

    try:
        if os.path.isdir(urlpath):
            files.extend(glob.glob(os.path.join(urlpath, '*.yaml')))
        elif os.path.isfile(urlpath):
            files.append(urlpath)
    except TypeError as e:
        rh_logger.logger.report_exception(exception=e,
                                          msg="Could not find valid files")

    for fname in files:
        rh_logger.logger.report_event("Loading from {}".format(fname))
        try:
            with open(fname, 'r') as f:
                meta = yaml.load(f)
                for model in meta:
                    pairs.append((meta[model]['url'], meta[model]['hash']))

        except (KeyError, FileNotFoundError, yaml.scanner.ScannerError) as e:
            # Skip over files that throw errors, recording file name and error
            rh_logger.logger.report_exception(exception=e,
                                              msg="URL could not be loaded from {}".format(fname))
            continue

    return pairs


def should_evaluate_model(db, url, md5_hash):
    """Determine if the model has been evaluated previously.

    Parameters
    ----------
    db : dict or str
        The in-memory dictionary copy of the database, or the path to the
        database file.
    url : str
        URL at which to download the model files.
    md5_hash : str
        Hash of the model file for comparison with the database.

    Returns
    -------
    True if the model has not previously been evaluated by the test harness,
    False otherwise.
    """
    rh_logger.logger.report_event('Determining if model is in database')
    for i in range(len(db)):
        if url == db[i]['url'] and md5_hash == db[i]['hash']:
            rh_logger.logger.report_event('Skipping model {}'.format(url))
            return False
    rh_logger.logger.report_event('Processing model {}'.format(url))
    return True


def download_model(url, md5_hash, tempdir):
    """Download a model and add it to the temprary workspace.

    Parameters
    ----------
    url : str
        URL at which to download the model files.
    md5_hash : str
        Hash of the model file for comparison with the downloaded file.
    tempdir : str
        Path to the temporary workspace.

    Returns
    -------
    """
    rh_logger.logger.report_event('Downloading model {}'.format(url))

    try:
        local_path = os.path.join(tempdir, os.path.basename(url))
        urllib.urlretrieve(url, local_path)
    except (urllib.URLError, HTTPError) as e:
        rh_logger.logger.report_exception(exception=e)
        return False
    else:
        local_hash = hashlib.md5(local_path).hexdigest()

    rh_logger.logger.report_event('Extracting model {}'.format(url))
    try:
        if local_hash == md5_hash:
            rh_logger.logger.report_event('Hash {} matches {}'.format(local_hash, md5_hash))
            if tarfile.is_tarfile(local_path):
                with tarfile.open(local_path, 'r') as t:
                    t.extractall(path=os.path.join(tempdir, 'test_subject'))
            elif zipfile.is_zipfile(local_path):
                with zipfile.ZipFile(local_path, 'r') as z:
                    z.extractall()
                    os.path.join(tempdir, 'test_subject')
    except IOError as e:
        rh_logger.logger.report_exception(exception=e)
        return False
    else:
        return True


def get_model_info(tempdir):
    """Extract information about the model from its YAML file.

    Parameters
    ----------
    tempdir : str
        Path to the temporary workspace.

    Returns
    -------
    info : tuple of (str, str, str)
        A tuple containing the path to the synapse classifier, the path to its
        model file, and the path to its weights file.
    """
    config_path = os.path.join(tempdir, 'local_path', 'classifier.yaml')

    try:
        with open(config_path, 'r') as f:
            config = yaml.load(f)
    except (IOError, TypeError) as e:
        rh_logger.logger.report_exception(exception=e)
        return

    try:
        padding = [int(config['classifier']['xy-pad-size']),
                   int(config['classifier']['xy-pad-size']),
                   int(config['classifier']['z-pad-size'])]
    except KeyError:
        padding = [0, 0, 0]

    try:
        classes = config['classifier']['classes']
    except KeyError:
        classes = ['transmitter', 'receptor']

    try:
        model_path = os.path.basename(config['classifier']['model-path'])
        weights_path = os.path.basename(config['classifier']['weights-path'])
    except KeyError as e:
        rh_logger.logger.report_exception(exception=e)
        return

    return (model_path,
            weights_path,
            padding,
            classes)


def update_paths(tempdir, aggregate, synapse, synapse_model, synapse_weights):
    """Update the classifiers to point to the correct paths.

    Parameters
    ----------
    tempdir : str
        Path to the temporary workspace.
    aggregate : str
        Name of the aggregate classifier pickle file in the temporary
        workspace.
    synapse : str
        Name of the synapse classifier pickle file in the temporary workspace.
    synapse_model : str
        Name of the synapse classifier model file in the temporary workspace.
    synapse_weights : str
        Name of the synapse classifier weights file in the temporary workspace.
    """
    with open(os.path.join(tempdir, 'classifiers', aggregate), 'r') as f:
        c = pickle.load(f)

    c.pickle_paths[1] = os.path.join(tempdir, 'test_subject', synapse)
    with open(os.path.join(tempdir, 'classifiers', aggregate), 'w') as f:
        pickle.dump(c, f)

    with open(os.path.join(tempdir, 'test_subject', synapse), 'r') as f:
        c = pickle.load(f)

    c.model_path = os.path.join(tempdir, 'test_subject', synapse_model)
    c.weights_path = os.path.join(tempdir, 'test_subject', synapse_weights)
    with open(os.path.join(tempdir, 'test_subject', synapse), 'w') as f:
        pickle.dump(c, f)


def run_pipeline(tempdir, padding, classes):
    """Run the ARIADNE pipeline on the test subject synapse classifier.

    Parameters
    ----------

    """
    args = [
        'bash',
        'run_ecs_test.sh',
        os.path.join(tempfile, 'results'),
        os.path.join(tempfile,
                     'classifiers',
                     '2017-05-04_membrane_2017-05-04_synapse.pkl'),
        str(padding[0] if padding[0] >= 0 and padding[0] < 145 else 0),
        str(padding[1] if padding[1] >= 0 and padding[1] < 1496 else 0),
        str(padding[2] if padding[2] >= 0 and padding[2] < 1496 else 0),
    ]

    if classes == ['transmitter', 'receptor']:
        args.append('--wants-transmitter-receptor-synapse-maps')

    subprocess.call(args, shell=True)


def format_results(tempdir, shape):
    """Format the segmentation results for friendly NRI calculation.

    Parameters
    ----------
    tempdir : str
        Path to the temporary workspace.
    shape : tuple of (int, int, int)
        The 3D shape of the raw image stack.
    """
    synapses = np.zeros(shape, dtype=np.uint16)
    presynaptic = np.zeros(shape, dtype=np.uint8)

    for root, dirs, files in os.walk(os.path.join(tempdir, 'results')):
        plans = glob.glob(os.path.join(root, '*.loading.plan'))
        if len(files) > 0 and len(plans) > 0:
            for plan in plans:
                if plan.find('synapse_segmentation') >= 0:
                    seg = DestVolumeReader(plan).imread()
                    offset = plan.split('_')[1:-1]
                    s = []
                    e = []
                    for i in offset:
                        idx = i.split('-')
                        s.append(idx[0])
                        e.append(idx[0])
                    synapses[s[2]:e[2], s[1]:e[1], s[0]:e[0]] += seg

                if plan.find('transmitter') >= 0:
                    seg = DestVolumeReader(plan).imread()
                    offset = plan.split('_')[1:-1]
                    s = []
                    e = []
                    for i in offset:
                        idx = i.split('-')
                        s.append(idx[0])
                        e.append(idx[0])
                    presynaptic[s[2]:e[2], s[1]:e[1], s[0]:e[0]] += seg

    outpath = os.path.join(tempdir, 'results', 'synapse_segmentation.h5')
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('stack', data=synapses)

    outpath = os.path.join(tempdir, 'results', 'presynaptic_map.h5')
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('stack', data=presynaptic)

    # outpath = os.path.join(tempdir, 'results', 'stitched_segmentation.h5')
    # with h5py.File(outpath, 'r+') as f:
    #     key = f.keys()[0]
    #     stitch = f[key][:]
    #     offset = list(map(lambda x: ()))
    #     seg = np.zeros(shape, dtype=stack.dtype)
    #     seg
    #     del f[key]
    #     f.create_dataset(key, data=seg)


def load_vi(tempdir):
    """Load VI results and reformat them to a dictionary.

    Parameters
    ----------
    tempdir : str
        Path to the temporary workspace.

    Returns
    -------
    vi : dict
        Dictionary containing VI statstics output by the ARIADNE pipeline.
    """
    vi = {}
    stats_path = os.path.join(tempdir,
                              'results',
                              'segmentation_statistics.csv')
    rows = []
    with open(stats_path, 'r') as f:
        reader = csv.reader(f)
        for row in reader:
            rows.append(row)
    for i in range(len(rows[0])):
        vi[rows[0][i]] = float(rows[1][i])

    return vi


def calculate_nri(tempfile, gt_dir):
    """Calculate the NRI from segmentation results.

    Parameters
    ----------
    tempdir : str
        Path to the temporary workspace.
    gt_dir : str
        Path to the directory containing ground truth.

    Returns
    -------
    nri : dict
        Dictionary containing NRI statstics.
    """
    nri = {}

    # Run the NRI Python script and collect the output
    args = [
        'python',
        'nri.py',
        '--segmentation-file',
        os.path.join(tempfile, 'results', 'stitched_segmentation.h5',),
        '--synapse-segmentation-file',
        os.path.join(tempfile, 'results', 'synapse_segmentation.h5',),
        '--pre-synaptic-map-file',
        os.path.join(tempfile, 'results', 'presynaptic_map.h5',),
        '--ground-truth-file',
        os.path.join(gt_dir, 'seg_groundtruth0.h5'),
        '--ground-truth-synapse-file',
        os.path.join(gt_dir, 'synapse_groundtruth.h5'),
    ]

    out = subprocess.check_output(args)

    # Iterate over each line and capture the precision, recall, and nri values
    for line in out.splitlines():
        if re.match(r'(Precision:)|(Recall:)|(NRI:)', line) is not None:
            words = line.strip().split()
            nri[words[0].strip(':').lower()] = float(words[-1])

    return nri


def update_database(db, url, md5_hash, nri, vi, time):
    entry = {
        'url': url,
        'hash': md5_hash,
        'stats': nri.copy()
    }
    entry['stats'].update(vi)
    entry['stats']['running_time'] = time

    db.append(entry)
    return db


def main():
    args = parse_args()

    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    tempdir = None

    with open(args.database, 'r') as f:
        db = json.load(f)

    with h5py.File(os.path.join(args.dataset, 'gt-4x6x6.h5'), 'r') as f:
        shape = f[f.keys()[0]].shape

    urls = load_urls()

    for url, md5_hash in urls:
        if should_evaluate_model(db, url, md5_hash):
            if tempdir is None:
                prep(args.aggregate_classifier,
                     args.membrane_classifier,
                     args.membrane_classifier_model,
                     args.membrane_classifier_weights)
            # Get the model and its information
            download_model(url, md5_hash, tempdir)
            classifier_info = get_model_info(tempdir)

            if classifier_info is None:
                continue


            update_paths(tempdir,
                         '2017-05-04_membrane_2017-05-04_synapse.pkl',
                         *classifier_info)

            run_pipeline()
            format_results(tempdir, shape)
            vi = load_vi(tempdir)
            nri = calculate_nri(tempdir)
            db = update_database(db, nri, vi)

    with open(args.database, 'w') as f:
        json.dump(db, f)

    rh_logger.logger.end_process("Evaluation complete")


if __name__ == '__main__':
    main()
