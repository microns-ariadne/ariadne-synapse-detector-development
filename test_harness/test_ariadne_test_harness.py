import pytest

import argparse
import os
import tempfile

from ariadne_test_harness import parse_args, prep, load_urls
from ariadne_test_harness import should_evaluate_model, download_model
from ariadne_test_harness import get_model_info, update_paths, run_pipeline
from ariadne_test_harness import format_results, load_vi, calculate_nri
from ariadne_test_harness import update_database, clean_up, main


def test_parse_args():
    args = parse_args()
    assert args == argparse.Namespace()

    args = parse_args(['-d', 'foo/bar.json'])
    assert isinstance(args.database, str)
    assert args.database == 'foo/bar.json'


def test_prep():
    # Ensure that the temporary directories are set up correctly
    temp = prep('/scratch0/test_harness_classifier')
    assert os.path.isdir(temp)
    assert os.path.isdir(os.path.join(temp, 'classifiers'))
    assert os.path.isdir(os.path.join(temp, 'results'))
    assert os.path.isdir(os.path.join(temp, 'test_subject'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '2017-05-04_membrane_2017-05-04_synapse.pkl'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '2017-05-04_membrane.pkl'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000.json'))
    assert os.path.isfile(os.path.join(temp, 'classifiers', '3D_unet_ecs_pixel_half_same_unnorm_380_12_47000_weights.h5'))

    os.rmdir(temp)

    # Test for exception in the case that the classifiers don't exdigest
    with pytest.raises(FileNotFoundError):
        temp = prep('/foo/bar')


def test_load_urls():
    # Test using a placeholder with two urls
    testpath = os.path.basename(__file__)
    pairs = load_urls(os.path.join(testpath, 'test_files/urls.yaml'))
    assert pairs[0] == ('test_files/ref/synapse1.tar.gz',
                        'f2546e932abd0ec3f678b900e2edcd2e')
    assert pairs[1] == ('test_files/ref/synapse2.tar.gz',
                        'aa67038de3a1a60aaff0d7345d99949e')

    # Test supplying a directory instead of a yaml file
    pairs = load_urls(os.path.join(testpath, 'test_files'))
    assert pairs[0] == ('test_files/ref/synapse1.tar.gz',
                        'f2546e932abd0ec3f678b900e2edcd2e')
    assert pairs[1] == ('test_filesref/synapse2.tar.gz',
                        'aa67038de3a1a60aaff0d7345d99949e')

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
    db = {}
    pairs = load_urls('test_files')

    # Test on a new model


def test_download_model():
    pass


def test_get_model_info():
    pass


def test_update_paths():
    pass


def test_run_pipeline():
    pass


def test_format_results():
    pass


def test_load_vi():
    pass


def test_calculate_nri():
    pass


def test_update_database():
    pass


def test_clean_up():
    pass


def test_main():
    pass
