import pytest

import argparse
import os
import shutil
import tempfile

from ariadne_test_harness import parse_args, prep, load_urls
from ariadne_test_harness import should_evaluate_model, download_model
from ariadne_test_harness import get_model_info, update_paths, run_pipeline
from ariadne_test_harness import format_results, load_vi, calculate_nri
from ariadne_test_harness import update_database, clean_up, main
from ariadne_test_harness import PrepFailedException


def setup():
    base = os.path.dirname(__file__)
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
                    '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    pairs = get_model_info(os.path.join(base, 'test_files'))
    return base, temp, pairs


def test_prep():
    # Ensure that the temporary directories are set up correctly
    base = os.path.join(os.path.dirname(__file__), 'test_files', 'prep')
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
                    '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5')
                )
    assert os.path.isdir(temp)
    assert os.path.isdir(os.path.join(temp, 'classifiers'))
    assert os.path.isdir(os.path.join(temp, 'results'))
    assert os.path.isdir(os.path.join(temp, 'test_subject'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '2017-05-04_membrane_2017-05-04_synapse.pkl'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '2017-05-04_membrane.pkl'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    os.rmtree(temp)

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

    pairs = load_urls(os.path.join(testpath, 'urls.yaml'))
    assert pairs[0] == ('test_files/download_model/synapse1.tar.gz',
                        'aa2abfe41ae4592d4505e28a04d96ffe')
    assert pairs[1] == ('test_files/download_model/synapse2.tar.gz',
                        '4aea190003303370f41c6775cd789ad5')

    # Test supplying a directory instead of a yaml file
    pairs = load_urls(testpath)
    assert pairs[0] == ('test_files/download_model/synapse1.tar.gz',
                        'aa2abfe41ae4592d4505e28a04d96ffe')
    assert pairs[1] == ('test_files/download_model/synapse2.tar.gz',
                        '4aea190003303370f41c6775cd789ad5')

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
    os.mkdir(os.path.join(temp, 'test_subject'))

    db = []
    db.append({'url': 'test_files/download_model/synapse1.tar.gz',
               'hash': 'aa2abfe41ae4592d4505e28a04d96ffe'})
    db.append({'url': 'test_files/download_model/synapse2.tar.gz',
               'hash': '4aea190003303370f41c6775cd789ad5'})

    # Test on one model
    assert download_model(db[0]['url'], db[0]['hash'], temp) is True
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        '2017-05-04_synapse.pkl'))
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        'classifier.yaml'))
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000.json'))
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000_weights.h5'))

    shutil.rmtree(os.path.join(temp, 'test_subject'))
    os.mkdir(os.path.join(temp, 'test_subject'))

    # Test on the other model
    assert download_model(db[1]['url'], db[1]['hash'], temp) is True
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        '2017-05-04_synapse.pkl'))
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        'classifier.yaml'))
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000.json'))
    assert os.path.isfile(os.path.join(
        temp,
        'test_subject',
        '3D_unet_ecs_synapse_polarity_pre_linear_316_32_115000_weights.h5'))

    shutil.rmtree(os.path.join(temp, test_subject))

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
    pass


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
        assert f[f.keys()[0]][:] == actual_synapse

    with h5py.File(os.path.join(base, 'presynaptic_map.h5'), 'r') as f:
        assert f[f.keys()[0]][:] == actual_presynaptic

    del actual_synapse
    del actual_presynaptic

    # Test on invalid directory
    temp = tempfile.mkdtemp()
    format_results(temp, (145, 1496, 1496))

    with h5py.File(os.path.join(temp, 'synapse_segmentation.h5'), 'r') as f:
        assert np.all(f[f.keys()[0]][:] == 0)

    with h5py.File(os.path.join(temp, 'presynaptic_map.h5'), 'r') as f:
        assert np.all(f[f.keys()[0]][:] == 0)


def test_load_vi():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'load_vi')

    # Test the standard calculation
    vi = load_vi(temp)
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
    f = open(os.path.join(temp, 'results', 'segmentation_statistics.csv'), 'w')
    f.close()
    vi = load_vi(temp)
    assert vi == {}

    # Test on an empty directory
    os.remove(os.path.join(temp, 'results', 'segmentation_statistics.csv'))
    vi = load_vi(temp)
    assert vi == {}


def test_calculate_nri():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'claculate_nri')

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
    os.remove(os.path.join(temp, 'results', 'stitched_segmentation.h5'))
    os.remove(os.path.join(temp, 'results', 'synapse_segmentation.h5'))
    os.remove(os.path.join(temp, 'results', , 'presynaptic_map.h5'))
    nri = calculate_nri()

    assert nri == {}


def test_update_database():
    base = os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        'test_files',
        'update_database')

    # Test adding actual data


def test_main():
    pass
