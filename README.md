# Instructions for Harvard lab install for classifier development

These are instructions for running the Ariadne pipeline on the R0 dataset
to produce evaluation metrics for classifiers.

## Installation

Installation is for Linux only, tested on Ubuntu 16.04. You must have
Anaconda installed. You must also have read access to the Github repos:

```
https://github.com/VCG/microns_skeletonization
https://github.com/microns-ariadne/pipeline_engine

```

These are the steps:
```
$ git clone git@github.com:microns-ariadne/pipeline_engine
$ cd pipeline_engine
$ conda env create -f conda-install.yaml
$ source activate ariadne_microns_pipeline
$ pip install --process-dependency-links .
$ pip install keras==1.2.1
$ pip install .
$ cd ..
$ git clone https://github.com/rhoana/butterfly
$ cd butterfly
$ pip install .
```

## Specifying your classifier

The script will pickle your classifier, which should consist of a Keras
model .json file and an HDF5 weights file. You have to rename
"example_classifier.yaml" to "classifier.yaml" (in this directory) and
then edit classifier.yaml with the details regarding your classifier. The
classifier.yaml file should be self-documenting.

## Running

You should have coxfs01 mounted on your system as /n/coxfs01 (if not, you'll
have to hand-edit lab-rh-config.yaml to change paths).

Set the following environment variables:

MICRONS_TMP_DIR - a directory for intermediate results and logfiles
MICRONS_ROOT_DIR - a directory for the report.json file

Do the following from this directory:
```
$ source activate ariadne_microns_pipeline
$ ./run_synapse.sh
```

An example pickle file is /n/coxfs01/leek/classifiers/2017-05-04/2017-05-04_synapse.pkl
Pickling a classifier is TODO. For now, start IPython and type

```
from ariadne_microns_pipeline.classifiers.keras_classifier import KerasClassifier
from ariadne_microns_pipeline.algorithms.normalize import NormalizeMethod
help(KerasClassifier)
k = KerasClassifier(<path-to-model-file>, <path-to-weights-file>...)
import cPickle
cPickle.dump(k, open(<path-to-pickle-file>, "w"))
```

## Troubleshooting

During the pipeline, Luigi is running on your computer at port 8082. Go to
http://localhost:8082 to see your progress.