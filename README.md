# ARIADNE Synapse Detector development

This repo contains tools and starter scripts for developing synapse detector
models for use with the ARIADNE pipeline, as well as a tool to upload trained
models to an evaluation framework.

## Installation

Installation is for Linux only, tested on Ubuntu 16.04 and RHEL 7. You must have
Anaconda installed. You must also have read access to the Github repos:

[MICRONS Skeletonization](https://github.com/VCG/microns_skeletonization)
[ARIADNE Pipeline](https://github.com/microns-ariadne/pipeline_engine)

To install these dependencies, run the following commands:

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

Then, run the following commands to install the tools in this repository:

```
$ git clone https://github.com/microns-ariadne/ariadne-synapse-detector-development
$ cd ariadne-synapse-detector-development
$ pip install --editable .
```

To verify installation, run `synapse-detector-development --help`, which should
display the possible commands to run.

## Creating A New Synapse Detector Project

To initialize a new project, run

```
synapse-detector-development create -n myproject $HOME
```

This will create a new directory `myproject` in your home directory with the
following files:

```
myproject
├── classifier.yaml
├── custom_layers.py
├── custom_layers.pyc
└── synapse_detector_training.py
```

## Developing A New Synapse Detector

The `synapse_detector_training.py` script contains sample code for developing
and training a new synapse detector using Keras. Develop your Keras model in
within the `create_model` function on line 54.

```python
def create_model(in_shape, out_shape):
    model = unet3d()
    model.compile(loss='cosine_proximity', optimizer='adam')
    return model
```

The example function here loads a reference 3D U-Net model from this package.
To develop your own model, just add your layers to the top of the function and
replace `model = unet3d()` with the standard Keras model instantiation.

## Training A Synapse Detector

The `main` function in `synapse_detector_training.py` contains code for loading
data and training the model.

```python
def main():
    # Define the input and output shapes of your model.
    INPUT_SHAPE = (17, 512, 512, 1)
    OUTPUT_SHAPE = (17, 512, 512, 1)

    model = create_model(INPUT_SHAPE, OUTPUT_SHAPE)
    raw, gt, dist = load()
    # The load function can also be provided paths manually, like so:
    # load(dataset=<path-to-raw-data>, gt=<path-to-gt)

    # Create the training data generator
    train_data = augmenting_generator(raw, gt, dist, INPUT_SHAPE,
                                      OUTPUT_SHAPE, 50)

    # Train the model with 50 batches per eposh over 50 epochs
    model.fit_generator(train_data, 50, 50, verbose=2)

    # Save the model to file. Rename the fiel as you see fit.
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights()
```

The line `raw, gt, dist = load()` will automatically load the synapse training
set from `coxfs`.

After training, be sure to save your model with

```python
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights()
```

as shown in the `main` function.

## Adding Custom Layers

Custom layers must be added to the `custom_layers.py` script in the project
directory. Add the custom layer class as usual under the `MyLayer` class, then
register the layer at the bottom to allow imports into other modules:

```python
    custom_objects = {
        'MyLayer': MyLayer,
    #   'custom_layer': custom_layer
    }

    __all__ = ['custom_objects',
               'MyLayer',
            #  'custom_layer'
               ]
```

This will enable the ARIADNE pipeline to reconstruct a model that uses the
custom layer.

For information about writing custom layers, see the [Keras documentation](https://faroit.github.io/keras-docs/1.2.2/layers/writing-your-own-keras-layers/)

## Updating the Metadata File

The `classifier.yaml` file contains a number of potential settings necessary
for the ARIADNE pipeline to reconstruct a trained Keras model. Ensure that
these settings are correct for your model.

# Uploading a Model

To upload a model, run the following command:

```
synapse-detector-development \
    <path-to-model.json> \
    <path-to-weights.h5> \
    <path-to-classifier.yaml> \
    [<path-to-custom_layers.py>, optional]
```

The command line will prompt you to name the model and provide other,
information, then the model will be uploaded to the evaluation pipeline server
to be processed.
