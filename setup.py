from setuptools import setup

with open('README.md', 'r') as f:
    README = f.read()

setup(
    name='synapse-detector-development',
    version='0.0.1',
    description='Tools and examples for synapse detector development.',
    long_description=README,
    url="https://github.com/jeffkinnison/syndetect",
    packages=['synapse_detector_development',
              'synapse_detector_development.commands',
              'synapse_detector_development.commands.samples',
              'synapse_detector_development.reference', ],
    include_package_data=True,
    install_requires=[
        'docopt',
        'h5py',
        'keras==1.2.2',
        'numpy',
        'pycurl',
        'pyyaml',
        'rh_config',
        'rh_logger',
        'scipy',
    ],
    dependency_links=[
        'https://github.com/Rhoana/rh_config/archive/1.0.0.tar.gz#egg=rh_config-1.0.0',
        'https://github.com/Rhoana/rh_logger/archive/2.0.0.tar.gz#egg=rh_logger-2.0.0',
    ],
    entry_points={
        'console_scripts': [
            "synapse-detector-development = synapse_detector_development.cli:main",
        ]
    },
)
