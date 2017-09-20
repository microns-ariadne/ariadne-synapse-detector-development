import pytest

import argparse
try:
    import cPickle as pickle
except ImportError:
    import pickle
import os
import shutil
import tempfile

import h5py
import numpy as np

from ariadne_test_harness import parse_args, prep, load_urls
from ariadne_test_harness import should_evaluate_model, download_model
from ariadne_test_harness import update_paths, run_pipeline
from ariadne_test_harness import format_results, load_vi, calculate_nri
from ariadne_test_harness import update_database, main
from ariadne_test_harness import PrepFailedException


def test_prep():
    # Ensure that the temporary directories are set up correctly
    base = os.path.join(os.path.dirname(__file__), 'test_files', 'prep')
    temp, classifiers, results, test_subject = \
        prep(os.path.join(
                base,
                '2017-05-04_membrane_2017-05-04_synapse.pkl'),
             os.path.join(
                base,
                '2017-05-04_membrane.pkl'),
             os.path.join(
                base,
                '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'),
             os.path.join(
                base,
                '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    aggregate = os.path.join(
        classifiers,
        '2017-05-04_membrane_2017-05-04_synapse.pkl')
    membrane = os.path.join(
        classifiers,
        '2017-05-04_membrane.pkl')
    membrane_model = os.path.join(
        classifiers,
        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json')
    membrane_weights = os.path.join(
        classifiers,
        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5')

    assert os.path.isdir(temp)
    assert os.path.isdir(classifiers)
    assert os.path.isdir(results)
    assert os.path.isdir(test_subject)
    assert os.path.isfile(aggregate)
    assert os.path.isfile(membrane)
    assert os.path.isfile(membrane_model)
    assert os.path.isfile(membrane_weights)

    with open(aggregate, 'r') as f:
        c = pickle.load(f)
    assert c.pickle_paths[0] == membrane

    with open(membrane, 'r') as f:
        c = pickle.load(f)
    assert c.model_path == membrane_model
    assert c.weights_path == membrane_weights

    shutil.rmtree(temp)

    # Test for exception in the case that the classifiers don't exist
    with pytest.raises(PrepFailedException):
        temp = prep(os.path.join(
                        base,
                        'foo.pkl'),
                    os.path.join(
                        base,
                        '2017-05-04_membrane.pkl'),
                    os.path.join(
                        base,
                        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'),
                    os.path.join(
                        base,
                        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    with pytest.raises(PrepFailedException):
        temp = prep(os.path.join(
                        base,
                        '2017-05-04_membrane_2017-05-04_synapse.pkl'),
                    os.path.join(
                        base,
                        'foo.pkl'),
                    os.path.join(
                        base,
                        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'),
                    os.path.join(
                        base,
                        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    with pytest.raises(PrepFailedException):
        temp = prep(os.path.join(
                        base,
                        '2017-05-04_membrane_2017-05-04_synapse.pkl'),
                    os.path.join(
                        base,
                        '2017-05-04_membrane.pkl'),
                    os.path.join(
                        base,
                        'foo.json'),
                    os.path.join(
                        base,
                        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    with pytest.raises(PrepFailedException):
        temp = prep(os.path.join(
                        base,
                        '2017-05-04_membrane_2017-05-04_synapse.pkl'),
                    os.path.join(
                        base,
                        '2017-05-04_membrane.pkl'),
                    os.path.join(
                        base,
                        '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'),
                    os.path.join(
                        base,
                        'foo.h5'))


def test_load_urls():
    # Test using a placeholder with two urls
    testpath = os.path.join(os.path.dirname(__file__),
                            'test_files',
                            'load_urls')

    syn1 = ('test_files/download_model/synapse1.tar.gz',
            'aa2abfe41ae4592d4505e28a04d96ffe')

    syn2 = ('test_files/download_model/synapse2.tar.gz',
            '4aea190003303370f41c6775cd789ad5')

    # Test with ideal input
    pairs = load_urls(os.path.join(testpath, 'urls.yaml'))
    assert syn1 in pairs
    assert syn2 in pairs

    # Test supplying a directory instead of a yaml file
    pairs = load_urls(testpath)
    assert syn1 in pairs
    assert syn2 in pairs

    # Test supplying a non-yaml file
    pairs = load_urls(__file__)
    assert pairs == []

    # Test supplying a non-existent yaml file
    pairs = load_urls(os.path.join(testpath, 'slkdgh.yaml'))
    assert pairs == []

    # Test supplying a yaml file that does not have the expected fields


    # Test supplying a directory with no yaml files
    temp = tempfile.mkdtemp()
    pairs = load_urls(temp)
    assert pairs == []
    os.rmdir(temp)

    # Test supplying a directory that does not exist
    pairs = load_urls(temp)
    assert pairs == []

    # Test supplying non-string values
    pairs = load_urls(7)
    assert pairs == []

    pairs = load_urls([])
    assert pairs == []

    pairs = load_urls(None)
    assert pairs == []


def test_should_evaluate_model():
    db = []
    pairs = [
        ('test_files/download_model/synapse1.tar.gz',
         'aa2abfe41ae4592d4505e28a04d96ffe'),
        ('test_files/download_model/synapse2.tar.gz',
         '4aea190003303370f41c6775cd789ad5')
    ]

    # Test on a new model
    assert should_evaluate_model(db, *pairs[0]) is True

    # Test on a second new model
    assert should_evaluate_model(db, *pairs[1]) is True

    # Test on existing models
    db.append({'url': pairs[0][0], 'hash': pairs[0][1]})
    assert should_evaluate_model(db, *pairs[0]) is False

    db.append({'url': pairs[1][0], 'hash': pairs[1][1]})
    assert should_evaluate_model(db, *pairs[1]) is False


def test_download_model():
    # Prepare the test
    temp = tempfile.mkdtemp()
    temp = os.path.join(temp, 'test_subject')
    os.mkdir(temp)

    db = []
    db.append({'url': 'test_files/download_model/synapse1.tar.gz',
               'hash': 'aa2abfe41ae4592d4505e28a04d96ffe'})
    db.append({'url': 'test_files/download_model/synapse2.tar.gz',
               'hash': '4aea190003303370f41c6775cd789ad5'})

    # Test on one model
    assert download_model(db[0]['url'], db[0]['hash'], temp) is True
    assert os.path.isfile(os.path.join(temp, 'synapse1.tar.gz'))
    assert os.path.isfile(os.path.join(
        temp,
        '2017-05-04_synapse.pkl'))
    assert os.path.isfile(os.path.join(
        temp,
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000.json'))
    assert os.path.isfile(os.path.join(
        temp,
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000_weights.h5'))

    shutil.rmtree(temp)
    os.mkdir(temp)

    # Test on the other model
    assert download_model(db[1]['url'], db[1]['hash'], temp) is True
    assert os.path.isfile(os.path.join(temp, 'synapse2.tar.gz'))
    assert os.path.isfile(os.path.join(
        temp,
        '2017-05-04_synapse.pkl'))
    assert os.path.isfile(os.path.join(
        temp,
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000.json'))
    assert os.path.isfile(os.path.join(
        temp,
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000_weights.h5'))

    shutil.rmtree(os.path.dirname(temp))

    # Test on invalid url
    assert download_model('file:///foo/bar.tgz', 'dummy', temp) is False

    # Test on invalid hash
    assert download_model(db[0]['url'], 'dummy', temp) is False


def test_update_paths():
    # Prepare the test
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'update_paths')

    aggregate = os.path.join(
        base,
        '2017-05-04_membrane_2017-05-04_synapse.pkl'
    )

    synapse = os.path.join(
        base,
        '2017-05-04_synapse.pkl'
    )

    synapse_model = os.path.join(
        base,
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000.json'
    )

    synapse_weights = os.path.join(
        base,
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000_weights.h5'
    )

    # Copy the original path information to restore it at the end of the test
    with open(aggregate, 'r') as f:
        c = pickle.load(f)
        orig_pkl_path = c.pickle_paths[1]

    with open(synapse, 'r') as f:
        c = pickle.load(f)
        orig_model_path = c.model_path
        orig_weights_path = c.weights_path

    del c

    # Update paths using ideal input
    s = update_paths(aggregate, synapse, synapse_model, synapse_weights)

    assert s is True

    with open(aggregate, 'r') as f:
        c = pickle.load(f)
        assert c.pickle_paths[1] == synapse

    with open(aggregate, 'w') as f:
        c.pickle_paths[1] = orig_pkl_path
        pickle.dump(c, f)

    with open(synapse, 'r') as f:
        c = pickle.load(f)
        assert c.model_path == synapse_model
        assert c.weights_path == synapse_weights

    with open(synapse, 'w') as f:
        c.model_path = orig_model_path
        c.weights_path = orig_weights_path
        pickle.dump(c, f)

    del c

    # Test on invalid classifiers
    s = update_paths(synapse, synapse, synapse_model, synapse_weights)
    assert s is False

    s = update_paths(aggregate, aggregate, synapse_model, synapse_weights)
    assert s is False

    # Test on non-pickle files
    s = update_paths(synapse_model, synapse, synapse_model, synapse_weights)
    assert s is False

    s = update_paths(aggregate, synapse_model, synapse_model, synapse_weights)
    assert s is False

    # Re-update the aggregate classifier test file
    with open(aggregate, 'r') as f:
        c = pickle.load(f)

    with open(aggregate, 'w') as f:
        c.pickle_paths[1] = orig_pkl_path
        pickle.dump(c, f)

    # Test on an empty directory
    s = update_paths(
            os.path.join('test_files', os.path.basename(aggregate)),
            os.path.join('test_files', os.path.basename(synapse)),
            os.path.join('test_files', os.path.basename(synapse_model)),
            os.path.join('test_files', os.path.basename(synapse_weights)))
    assert s is False


def test_run_pipeline():
    base = os.path.join(
        os.path.dirname(__file__),
        'test_files',
        'run_pipeline')
    temp = tempfile.mkdtemp()
    aggregate = os.path.join(
        base,
        '2017-05-04_membrane_2017-05-04_synapse.pkl')
    neuroproof = os.path.join(base, 'neuroproof', 'agglo_classifier_itr1.h5')
    rh_config = os.path.join(base, '.rh-config.yaml')

    # Test on the ideal case
    assert run_pipeline(temp, aggregate, neuroproof, rh_config) is True
    assert os.path.isfile(os.path.join(temp, 'stitched_segmentation.h5'))
    assert os.path.isfile(os.path.join(temp, 'segmentation-statistics.csv'))

    shutil.rmtree(temp)

    # Test with invalid results path
    invalid = '/foo/bar'
    assert run_pipeline(temp, aggregate, neuroproof, rh_config) is False

    # Test with invalid aggregate classifier
    assert run_pipeline(temp, neuroproof, neuroproof, rh_config) is False

    # Test with invalid neuroproof classifier
    assert run_pipeline(temp, aggregate, aggregate, rh_config) is False


def test_format_results():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'format_results')

    actual = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'calculate_nri')

    with h5py.File(os.path.join(actual, 'synapse_segmentation.h5'), 'r') as f:
        actual_synapse = f[f.keys()[0]][:]

    with h5py.File(os.path.join(actual, 'presynaptic_map.h5'), 'r') as f:
        actual_presynaptic = f[f.keys()[0]][:]

    # Test the ideal case
    format_results(base, (145, 1496, 1496))

    with h5py.File(os.path.join(base, 'synapse_segmentation.h5'), 'r') as f:
        assert np.array_equal(f[f.keys()[0]][:], actual_synapse)

    with h5py.File(os.path.join(base, 'presynaptic_map.h5'), 'r') as f:
        assert np.array_equal(f[f.keys()[0]][:], actual_presynaptic)

    os.remove(os.path.join(base, 'synapse_segmentation.h5'))
    os.remove(os.path.join(base, 'presynaptic_map.h5'))

    del actual_synapse
    del actual_presynaptic

    # Test on invalid directory
    temp = tempfile.mkdtemp()
    format_results(temp, (145, 1496, 1496))

    with h5py.File(os.path.join(temp, 'synapse_segmentation.h5'), 'r') as f:
        assert np.all(f[f.keys()[0]][:] == 0)

    with h5py.File(os.path.join(temp, 'presynaptic_map.h5'), 'r') as f:
        assert np.all(f[f.keys()[0]][:] == 0)

    shutil.rmtree(temp)


def test_load_vi():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'load_vi')

    # Test the standard calculation
    vi = load_vi(os.path.join(base, 'segmentation-statistics.csv'))
    assert vi['F_info'] == 0.9180095670059799
    assert vi['F_info_merge'] == 0.8743014354541885
    assert vi['F_info_split'] == 0.966317785639555
    assert vi['depth'] == 117
    assert vi['height'] == 1320
    assert vi['rand'] == 0.9084497848221688
    assert vi['rand_merge'] == 0.8533266740385526
    assert vi['rand_split'] == 0.9711863854029794
    assert vi['vi'] == 1.1004256256750584
    assert vi['vi_merge'] == 0.21473131776568866
    assert vi['vi_split'] == 0.8856943079093699
    assert vi['width'] == 1320
    assert vi['x'] == 88
    assert vi['y'] == 88
    assert vi['z'] == 14

    # Test on invalid csv file
    vi = load_vi(os.path.join(base, 'segmentation_statistics.csv'))
    assert vi == {}

    # Test on non-existent file
    vi = load_vi(os.path.join(base, 'foo.csv'))
    assert vi == {}


def test_calculate_nri():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'calculate_nri')

    # Test on the standard case
    nri = calculate_nri(
        base,
        os.path.join(base, 'seg_groundtruth0.h5'),
        os.path.join(base, 'synapse_groundtruth.h5'))

    assert 'nri' in nri
    assert nri['nri'] == 0.110
    assert 'precision' in nri
    assert nri['precision'] == 0.064
    assert 'recall' in nri
    assert nri['recall'] == 0.400

    # Test when results do not exist
    nri = calculate_nri(
        '/tmp',
        os.path.join(base, 'seg_groundtruth0.h5'),
        os.path.join(base, 'synapse_groundtruth.h5'))
    assert nri == {}

    # Test when ground truth does not exist
    nri = calculate_nri(
        base,
        os.path.join('/tmp', 'seg_groundtruth0.h5'),
        os.path.join(base, 'synapse_groundtruth.h5'))
    assert nri == {}

    nri = calculate_nri(
        '/tmp',
        os.path.join(base, 'seg_groundtruth0.h5'),
        os.path.join('/tmp', 'synapse_groundtruth.h5'))
    assert nri == {}


def test_update_database():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'update_database')

    # Test adding actual data


def test_main():
    pass
