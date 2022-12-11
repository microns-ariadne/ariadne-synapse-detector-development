import argparse
import csv
import glob
import hashlib
import json
import os
try:
    import cPickle as pickle
except ImportError:
    import pickle
import re
import shutil
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


class PrepFailedException(Exception):
    pass


def parse_args(args=None):
    p = argparse.ArgumentParser()

    p.add_argument('--url-files',
                   help='path to model url file or directory',
                   type=str)

    p.add_argument('--membrane-gt',
                   help='path to membrane ground truth',
                   type=str)

    p.add_argument('--synapse-gt',
                   help='path to synapse ground truth',
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

    p.add_argument('--database',
                   help='path to the JSON database file',
                   type=str)

    p.add_argument('--rh-config',
                   help='path to the .rh-config.yaml file for the pipeline',
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
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    rh_logger.logger.report_event("Preparing workspace")

    # Set up a temporary workspace
    tempdir = tempfile.mkdtemp(suffix='_ariadne_test_harness')
    rh_logger.logger.report_event(
        'Setting up temporary directories in {}'.format(tempdir))
    temp_classifiers = os.path.join(tempdir, 'classifiers')
    temp_results = os.path.join(tempdir, 'results')
    temp_test_subject = os.path.join(tempdir, 'test_subject')

    try:
        os.mkdir(temp_classifiers)
        os.mkdir(os.path.join(tempdir, 'results'))
        os.mkdir(os.path.join(tempdir, 'test_subject'))
    except OSError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg="Could not create directories")
        shutil.rmtree(tempdir)
        raise PrepFailedException

    # Copy the aggregate classifier, membrane classifier, membrane classifier
    # model file, and membrane classifier weights file to the temp workspace
    temp_agg = os.path.join(temp_classifiers, os.path.basename(agg))
    temp_membrane = os.path.join(temp_classifiers, os.path.basename(membrane))
    temp_model = os.path.join(temp_classifiers, os.path.basename(model))
    temp_weights = os.path.join(temp_classifiers, os.path.basename(weights))

    try:
        rh_logger.logger.report_event('Copying aggregate classifier')
        shutil.copy(os.path.abspath(agg), temp_agg)

        rh_logger.logger.report_event('Copying membrane classifier')
        shutil.copy(os.path.abspath(membrane), temp_membrane)

        rh_logger.logger.report_event('Copying membrane classifier model file')
        shutil.copy(os.path.abspath(model), temp_model)

        rh_logger.logger.report_event(
            'Copying membrane classifier weights file')
        shutil.copy(os.path.abspath(weights), temp_weights)

    except IOError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg="Could not copy classifier files")
        shutil.rmtree(tempdir)
        raise PrepFailedException

    # Set the filepaths in the aggregate and membrane classifiers to point to
    # the temp workspace
    try:
        with open(temp_agg, 'r') as f:
            c = pickle.load(f)
        c.pickle_paths[0] = temp_membrane
        with open(temp_agg, 'w') as f:
            pickle.dump(c, f)

        with open(temp_membrane, 'r') as f:
            c = pickle.load(f)
        c.model_path = temp_model
        c.weights_path = temp_weights
        with open(temp_membrane, 'w') as f:
            pickle.dump(c, f)
    except pickle.PickleError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg='File could not be pickled.')
        shutil.rmtree(tempdir)
        raise PrepFailedException

    return (tempdir, temp_classifiers, temp_results, temp_test_subject)


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
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    rh_logger.logger.report_event(
        "Loading URL/hash pairs from {}".format(urlpath))
    pairs = []
    files = []

    try:
        if os.path.isdir(urlpath):
            files.extend(glob.glob(os.path.join(urlpath, '*.yaml')))
        elif os.path.isfile(urlpath):
            files.append(urlpath)
    except TypeError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg="Could not find valid files")

    for fname in files:
        rh_logger.logger.report_event("Loading from {}".format(fname))
        try:
            with open(fname, 'r') as f:
                meta = yaml.load(f)
                for model in meta:
                    pairs.append((meta[model]['url'], meta[model]['hash']))

        except (KeyError, IOError, yaml.scanner.ScannerError) as e:
            # Skip over files that throw errors, recording file name and error
            rh_logger.logger.report_exception(
                exception=e,
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
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    rh_logger.logger.report_event('Determining if model is in database')
    for i in range(len(db)):
        try:
            if url == db[i]['url'] and md5_hash == db[i]['hash']:
                rh_logger.logger.report_event('Skipping model {}'.format(url))
                return False
        except KeyError as e:
            rh_logger.logger.report_exception(
                exception=e,
                msg='DB entry {} formatted incorrectly'.format(i))
            continue
        except IndexError as e:
            rh_logger.logger.report_exception(
                exception=e,
                msg='DB has no entry {}'.format(i))
            return False

    rh_logger.logger.report_event('Processing model {}'.format(url))
    return True


def download_model(url, md5_hash, temp_test_subject):
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
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    rh_logger.logger.report_event('Downloading model {}'.format(url))

    if os.path.isfile(url):
        url = ''.join(['file://', os.path.abspath(url)])

    try:
        local_path = os.path.join(temp_test_subject, os.path.basename(url))
        urllib.urlretrieve(url, local_path)
    except IOError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg='Could not download {}'.format(url))
        return False
    else:
        f = open(local_path, 'r')
        local_hash = hashlib.md5(f.read()).hexdigest()
        f.close()

    rh_logger.logger.report_event('Extracting model {}'.format(url))
    try:
        if local_hash == md5_hash:
            rh_logger.logger.report_event(
                'Hash {} matches {}'.format(local_hash, md5_hash))
            if tarfile.is_tarfile(local_path):
                with tarfile.open(local_path, 'r') as t:
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
                        
                    
                    safe_extract(t, path=temp_test_subject)
            elif zipfile.is_zipfile(local_path):
                with zipfile.ZipFile(local_path, 'r') as z:
                    z.extractall()
                    os.path.join(temp_test_subject)
    except IOError as e:
        rh_logger.logger.report_exception(exception=e)
        return False
    else:
        return True


def update_paths(aggregate, synapse, synapse_model, synapse_weights):
    """Update the classifiers to point to the correct paths.

    Parameters
    ----------
    aggregate : str
        Path to the aggregate classifier pickle file.
    synapse : str
        Path to the synapse classifier pickle.
    synapse_model : str
        Path to the synapse classifier model file.
    synapse_weights : str
        Path to the synapse classifier weights file.
    """
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    try:
        success = True
        with open(aggregate, 'r') as f:
            c = pickle.load(f)
        c.pickle_paths[1] = synapse
        with open(aggregate, 'w') as f:
            pickle.dump(c, f)

        with open(synapse, 'r') as f:
            c = pickle.load(f)
        if hasattr(c, 'model_path') and hasattr(c, 'weights_path'):
            c.model_path = synapse_model
            c.weights_path = synapse_weights
        else:
            raise AttributeError
        with open(synapse, 'w') as f:
            pickle.dump(c, f)
    except IOError as e:
        success = False
        rh_logger.logger.report_exception(exception=e,
                                          msg='Error opening file.')
    except pickle.UnpicklingError as e:
        success = False
        rh_logger.logger.report_exception(exception=e,
                                          msg='Error loading file.')
    except pickle.PicklingError as e:
        success = False
        rh_logger.logger.report_exception(exception=e,
                                          msg='Error pickling file.')

    except AttributeError as e:
        success = False
        rh_logger.logger.report_exception(exception=e,
                                          msg='Invalid classifier object')

    return success


def run_pipeline(temp_results, aggregate, neuroproof, rh_config):
    """Run the ARIADNE pipeline on the test subject synapse classifier.

    Parameters
    ----------

    """
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    args = [
        'bash',
        os.path.join(os.path.dirname(__file__), 'run_ecs_test.sh'),
        temp_results,
        aggregate,
        neuroproof,
        rh_config
    ]

    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg='Pipeline did not complete segmentation')
        return False
    else:
        return True


def format_results(temp_results, shape):
    """Format the segmentation results for friendly NRI calculation.

    Parameters
    ----------
    tempdir : str
        Path to the temporary workspace.
    shape : tuple of (int, int, int)
        The 3D shape of the raw image stack.
    """
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    synapses = np.zeros(shape, dtype=np.uint16)
    presynaptic = np.zeros(shape, dtype=np.uint8)

    for root, dirs, files in os.walk(temp_results):
        plans = glob.glob(os.path.join(root, '*.loading.plan'))
        if len(files) > 0 and len(plans) > 0:
            for plan in plans:
                if plan.find('synapse-segmentation') >= 0:
                    seg = DestVolumeReader(plan).imread()
                    offset = os.path.basename(plan).split('_')[1:-1]
                    s = []
                    e = []
                    for i in offset:
                        idx = i.split('-')
                        s.append(int(idx[0]))
                        e.append(int(idx[1]))
                    synapses[s[2]:e[2], s[1]:e[1], s[0]:e[0]] += seg

                if plan.find('transmitter') >= 0:
                    seg = DestVolumeReader(plan).imread()
                    offset = os.path.basename(plan).split('_')[1:-1]
                    s = []
                    e = []
                    for i in offset:
                        idx = i.split('-')
                        s.append(int(idx[0]))
                        e.append(int(idx[1]))
                    presynaptic[s[2]:e[2], s[1]:e[1], s[0]:e[0]] += seg

    outpath = os.path.join(temp_results, 'synapse_segmentation.h5')
    with h5py.File(outpath, 'w') as f:
        f.create_dataset('stack', data=synapses)

    outpath = os.path.join(temp_results, 'presynaptic_map.h5')
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


def load_vi(stats_file):
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
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    rh_logger.logger.report_event(
        'Loading VI information from {}'.format(stats_file))
    vi = {}
    rows = []
    try:

        with open(stats_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                rows.append(row)
        for i in range(len(rows[0])):
            rh_logger.logger.report_event(
                'Adding {} with value {}'.format(rows[0][i], rows[1][i]))
            vi[rows[0][i]] = float(rows[1][i])
    except IOError as e:
        rh_logger.logger.report_exception(exception=e,
                                          msg='Results file does not exist')
    except IndexError as e:
        rh_logger.logger.report_exception(exception=e,
                                          msg='Invalid csv file format')

    rh_logger.logger.report_event(
        'Loaded VI metrics {}'.format(vi))
    return vi


def calculate_nri(temp_results, membrane_gt, synapse_gt):
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
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    rh_logger.logger.report_event('Calculating NRI information')

    nri = {}

    # Run the NRI Python script and collect the output
    rh_logger.logger.report_event('Running nri.py')
    args = [
        'python',
        'nri.py',
        '--segmentation-file',
        os.path.join(temp_results, 'stitched_segmentation.h5',),
        '--synapse-segmentation-file',
        os.path.join(temp_results, 'synapse_segmentation.h5',),
        '--pre-synaptic-map-file',
        os.path.join(temp_results, 'presynaptic_map.h5',),
        '--ground-truth-file',
        os.path.join(membrane_gt),
        '--ground-truth-synapse-file',
        os.path.join(synapse_gt),
    ]

    try:
        out = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as e:
        out = ''
        rh_logger.logger.report_exception(exception=e,
                                          msg='nri.py failed to run')

    # Iterate over each line and capture the precision, recall, and nri values
    rh_logger.logger.report_event('Captured output {}'.format(out))
    for line in out.splitlines():
        if re.match(r'(Precision:)|(Recall:)|(NRI:)', line) is not None:
            words = line.strip().split()
            rh_logger.logger.report_event(
                'Found result {} {}'.format(words[0], words[1]))
            nri[words[0].strip(':').lower()] = float(words[1])

    if 'precision' in nri:
        nri['precision'] = nri['precision'] / 100.0
    if 'recall' in nri:
        nri['recall'] = nri['recall'] / 100.0

    rh_logger.logger.report_event('Captured NRI values {}'.format(nri))

    return nri


def update_database(db, url, md5_hash, nri, vi, time):
    """Update the database with new results.

    Record NRI, VI, and running time information into the database. If these
    values were not computed due to some error, skip this process.

    Parameters
    ----------
    db : dict
        The existing dictionary to append results to.
    url : str
        The URL at which the model can be found.
    md5_hash : str
        The MD5 hash of the model tar archive.
    nri : dict
        Statistics corresponding to the NRI measure.
    vi : dict
        Statistics corresponding to the VI measure.
    time : int or float
        The running time of the pipeline evaluation.

    Returns
    -------
    db : dict
        The updated database. If results were not added, equivalent to the `db`
        parameter.
    """
    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    if len(nri) > 0 and len(vi) > 0:
        entry = {
            'url': url,
            'hash': md5_hash,
            'stats': nri.copy()
        }
        entry['stats'].update(vi)
        entry['stats']['running_time'] = time
        db.append(entry)
    return db


def main(args=None):
    args = parse_args(args=args)

    try:
        rh_logger.logger.start_process("test_harness", "evaluate", [])
    except:
        pass

    tempdir = None
    aggregate_classifier = None

    with open(args.database, 'r') as f:
        db = json.load(f)

    with h5py.File(args.dataset, 'r') as f:
        shape = f[f.keys()[0]].shape

    urls = load_urls(args.urls)

    for url, md5_hash in urls:
        if should_evaluate_model(db, url, md5_hash):
            if tempdir is None:
                try:
                    tempdir, classifiers, results, test_subject = \
                        prep(args.aggregate_classifier,
                             args.membrane_classifier,
                             args.membrane_classifier_model,
                             args.membrane_classifier_weights)
                    aggregate_classifier = os.path.join(
                        classifiers,
                        os.path.basename(args.aggregate_classifier))
                except PrepFailedException:
                    break
            res = glob.glob(os.path.join(results, '*'))
            for f in res:
                if os.path.isdir(f):
                    shutil.rmtree(f)
                elif os.path.isfile(f):
                    os.remove(f)
            # Get the model and its information
            if not download_model(url, md5_hash, test_subject):
                continue

            synapse = glob.glob(os.path.join(test_subject, '*.pkl'))[0]
            synapse_model = glob.glob(os.path.join(test_subject, '*.json'))[0]
            synapse_weights = glob.glob(os.path.join(test_subject, '*.h5'))[0]

            update_paths(
                aggregate_classifier,
                synapse,
                synapse_model,
                synapse_weights,)

            run_pipeline(results, synapse, args.rh_config)
            format_results(results, shape)
            vi = load_vi(os.path.join(results, 'segmentation-statistics.csv'))
            nri = calculate_nri(results, args.membrane_gt, args.synapse_gt)
            db = update_database(db, nri, vi)

    try:
        shutil.rmtree(temp)
    except OSError as e:
        rh_logger.logger.report_exception(
            exception=e,
            msg='Could not remove temporary workspace at {}.'.format(temp))

    with open(args.database, 'w') as f:
        json.dump(db, f)

    rh_logger.logger.end_process("Evaluation complete")


if __name__ == '__main__':
    main()
