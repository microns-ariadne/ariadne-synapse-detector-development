import argparse
import json
import glob
import os

import h5py
import scipy.ndimage

import rh_config


class InvalidFileFormatError(Exception):
    def __init__(self, path):
        msg = 'The data at {} is not a recognized file format.'.format(path)
        super(InvalidFileFormatError, self).__init__(msg)


def calculate_distance(gt):
    """Compute the entry-wise distance to nonzeros in the ground truth.

    Parameters
    ----------
    gt : array-like
        Array containing the ground truth data.

    Returns
    -------
    distance : array-like
        Array containing the distance of each 0-valued entry in ``gt`` to the
        nearest non-zero entry.

    Notes
    -----
    The output of this function is used to generate training examples that have
    """
    return scipy.ndimage.distance_transform_edt(gt == 0, (30, 4, 4))


def load_from_image(path):
    """Load data from a directory of images.

    Parameters
    ----------
    path : str
        Path to the image directory.

    Returns
    -------
    vol : array-like
        The 3D array containing all of the image data in ``path`` in sequence.

    Notes
    -----
    This function assumes that 1) ``path`` is a directory, 2) ``path`` only
    contains images, and 3) the images in ``path`` will be in the correct order
    when their filenames are sorted.
    """
    files = sorted(glob.glob(os.path.join(path, '*')))
    vol = None
    for i in range(len(files)):
        img = scipy.ndimage.imread(files[i])
        if vol is None:
            vol = np.zeros(
                (len(files), img.shape[0], img.shape[1]),
                dtype=img.dtype)
        vol[i] += img

    return vol


def load_from_json(path):
    """Load data from a JSON specification.

    The JSON specification should contain three keys:

    1. "filename", the path to an HDF5 file with the image data
    2. "dataset-path", the path inside the HDF5 file to the data
    3. "z-offset", the offset from the top of the volume

    Parameters
    ----------
    path : str
        Path to the JSON specification file.

    Returns
    -------
    vol : array-like
        The image volume loaded from this JSON specification.
    """
    with open(path, 'r') as f:
        meta = json.load(f)

    if 'filename' in meta and 'dataset-path' in meta:
        vol = load_from_hdf5(meta['filename'], key=meta['dataset-path'])
        return vol[meta['z-offset']:]


def load_from_hdf5(path, key=None):
    """Load data from an HDF5 file.

    Parameters
    ----------
    path : str
        Path to the HDF5 file containing the data.
    key : str, optional
        The path to the data within the HDF5 file.

    Returns
    -------
    vol : array-like
        The image volume loaded from this JSON specification.

    Notes
    -----
    If ``key`` is not passed, this function will attempt to load the dataset
    from the first key in the HDF5 file.
    """
    with h5py.File(path, 'r') as f:
        try:
            x = f[key][:]
        except (KeyError, TypeError):
            x = f[f.keys()[0]][:]

    return x


def load_data(path):
    """Load data from a provided path.

    This function determines how to load data based on the file type and calls
    the appropriate function.

    Parameters
    ----------
    path : str
        File path to load data from.

    Returns
    -------
    vol : array-like
        The image volume loaded from this JSON specification.

    Raises
    ------

    """
    _, ext = os.path.splitext(path)

    if ext == '.json':
        vol = load_from_json(path)
    elif ext == '.h5':
        vol = load_from_hdf5(path)
    elif os.path.isdir(path):
        vol = load_from_image(path)
    else:
        raise InvalidFileFormatError

    return vol


def load(dataset=None, gt=None):
    """Load a dataset and associated ground truth.

    In the event that the optional parameters are not passed, this function
    attempts to load the ECS_iarpa_201610_gt_4x6x6 dataset from a
    .rh-config.yaml file.

    Parameters
    ----------
    dataset : str, optional
        Path to the dataset to load.
    gt : str, optional
        Path to the ground truth to load.

    Returns
    -------
    raw : `numpy.ndarray`
        The raw training data.
    gt : `numpy.ndarray`
        The ground truth to train on.
    dist : `numpy.ndarray`
        Array where each element is the distance from the corresponding element
        in ``gt`` to the nearest non-zerovalue in ``gt``.
    """
    # Load the dataset from rh_config if none provided.
    if dataset is None:
        experiments = rh_config.config['bfly']['experiments']
        for exp in experiments:
            if exp['name'] == 'ECS_train_images':
                for d in exp['datasets']:
                    if d['name'] == 'sem':
                        for channel in d['channels']:
                            if channel['name'] == 'raw':
                                dataset = channel['path']

    # Load the ground truth from rh_config if none provided.
    if gt is None:
        experiments = rh_config.config['bfly']['experiments']
        for exp in experiments:
            if exp['name'] == 'ECS_train_images':
                for d in exp['datasets']:
                    if d['name'] == 'sem':
                        for channel in d['channels']:
                            if channel['name'] == 'gt':
                                gt = channel['path']

    raw = load_data(dataset)
    gt = load_data(gt)

    dist = calculate_distance(gt)

    return (raw, gt, dist)
