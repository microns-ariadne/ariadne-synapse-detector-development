# ARIADNE Synapse Detector development

This repo contains tools and starter scripts for developing synapse detector
models for use with the ARIADNE pipeline, as well as a tool to upload trained
models to an evaluation framework.

# Table of Contents

[Dependencies](#dependencies)

[Installation](#installation)

[Creating a New Synapse Detector Project](#creating-a-new-synapse-detector-project)

[Developing a New Synapse Detector](#developing-a-new-synapse-detector)

[Training a Synapse Detector](#training-a-synapse-detector)

[Adding Custom Layers (Optional)](#adding-custom-layers-optional)

[Evaluating a Model Locally](#evaluating-a-model-locally)

[Updating the Metadata File](#updating-the-metadata-file)

[Uploading a Model](#uploading-a-model)

[Incorporating an Existing Model](#incorporating-an-existing-model)

# Dependencies

Installation is for Linux only, tested on Ubuntu 16.04 and RHEL 7. You must have
Anaconda installed. You must also have access to the following GitHub repos:

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

# Installation

```
  # Activate the ARIADNE pipeline conda environment if you have not
$ source activate ariadne_microns_pipeline
$ git clone https://github.com/microns-ariadne/ariadne-synapse-detector-development
$ cd ariadne-synapse-detector-development
$ conda install pycurl
$ pip install --editable .
```

To verify installation, run `synapse-detector-development --help`. If installed
correctly, this will display the possible commands to run.

```
ARIADNE Synapse Detector Development.

Tools for synapse detector development for MICrONS Team 1 TA3.

Usage:
    synapse-detector-development create [--metadata-only] [<name> <path>]
    synapse-detector-development upload <model-file> <weights-file> <metadata> <custom-layer-file>
    synapse-detector-development pickle <metadata>
    synapse-detector-development evaluate <model-file> <weights-file> <metadata> [<custom-layer-file>] [<rh-config>]
    synapse-detector-development -h | --help
    synapse-detector-development --version

Options:
    -h --help   Show this screen.
    --version   Show version.

Commands for synapse detector development are:
    create    Initialize a new synapse detector project with example files.
    evaluate  Run a trained classifier through the ARIADNE pipeline.
    upload    Submit a synapse detector model for evaluation.
```

# Creating a New Synapse Detector Project

To initialize a new project, run

```
$ synapse-detector-development create myproject $HOME
```

This will create a new directory `myproject` in your home directory. To change
the project location, replace `$HOME` with a different filepath. The
`myproject` directory contains the following files:

```
myproject
├── classifier.yaml
├── custom_layers.py
├── lab-rh-config.yaml 
└── synapse_detector_training.py
```

# Developing a New Synapse Detector

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
To develop your own model, add  to the top of the function and
replace `model = unet3d()` with the standard Keras model instantiation.

# Training A Synapse Detector

The `main` function in `synapse_detector_training.py` contains code for loading
data and training the model.

```python
def main():
    # Define the input and output shapes of your model.
    INPUT_SHAPE = (1, 32, 316, 316)
    OUTPUT_SHAPE = (1, 32, 316, 316)

    model = create_model(INPUT_SHAPE, OUTPUT_SHAPE)
    raw, gt, dist = load()
    print(raw.shape)
    # The load function can also be provided paths manually, like so:
    # load(dataset=<path-to-raw-data>, gt=<path-to-gt)

    # Create the training data generator
    x = np.expand_dims(np.copy(raw[56:-57, 590:-590, 590:-590]), axis=0)
    x = np.expand_dims(x, axis=0).astype(np.uint8)
    y = np.expand_dims(np.copy(gt[56:-57, 592:-592, 592:-592]), axis=0)
    y = np.expand_dims(y, axis=0).astype(np.uint8)

    # Train the model with 50 batches per eposh over 50 epochs
    model.fit(x, y, 1, 1, verbose=2)

    # Save the model to file. Rename the fiel as you see fit.
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights('weights.h5')
```

The line `raw, gt, dist = load()` will automatically load the synapse training
set from `coxfs`.

After training, your model will be saved by the `main` function with

```python
    with open('model.json', 'w') as f:
        json.dump(model.to_json(), f)

    model.save_weights()
```

This outputs `model.json`, which contains the model structure, and `weights.h5`,
which will be submitted for evaluation.

# Adding Custom Layers (Optional)

In the event that your model uses custom layers, the custom layer code must be
added to the `custom_layers.py` script in the project directory. Add the custom
layer class as usual under the `MyLayer` class, then register the layer at the
bottom to allow imports into other modules:

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

# Evaluating a Model Locally

Once the model is trained, run a local evaluation using the [ARIADNE pipeline](https://github.com/microns-ariadne/pipeline_engine) with

```
$ synapse-detector-development evaluate <model-file> <weights-file> <metadata-file> <custom-layers-file>
```

# Updating the Metadata File

The `classifier.yaml` file contains a number of potential settings necessary
for the ARIADNE pipeline to reconstruct a trained Keras model (e.g., padding for
the model output, the paths to your `model.json` and `weights.h5` files, etc.).

In most cases, the following fields will need to be updated:

* `model-path`: path to the `model.json` file
* `weights-path`: path to the `weights.h5` file
* `block-size`: list of batch block input dimesions `[z, y, x]`
* `xy-pad-size`: padding in the x- and y-dimensions to make model output shape match block input shape
* `z-pad-size`: padding in the z-dimension to make model output shape match block input shape

More information about these fields may be found in the `classifier.yaml` file.

# Uploading a Model

After the model has been trained and locally evaluated to satisfaction, run the
following command to submit it to the evaluation framework server:

```
synapse-detector-development upload \
    <path-to-model.json> \
    <path-to-weights.h5> \
    <path-to-classifier.yaml> \
    [<path-to-custom_layers.py>, optional]
```

The command line will prompt you to name the model and provide other,
information, then the model will be uploaded to the evaluation pipeline server
to be processed.

# Incorporating an Existing Model

Existing Keras models may be prepared for submission by following these steps:

1. Train the model and save the structure and weights as follows:
    
    ```python
    import json
    # Filenames may be replaced with something more meaningful
    json.dump(model.to_json(), open('model.json', 'w'))
    model.save_weights('weights.h5')
    ```

2. Run `synapse-detector-development create --copy-metadata` from the directory containing the model and weights files from the previous step.
3. Update `classifier.yaml` as described [here](#updating-the-metadata-file)
4. (optional) Run `synapse-detector-development create --copy-custom-layers` and update `custom_layers.py` as described [here](#adding-custom-layers-optional)
5. Submit the model as described [here](#uploading-a-model)
