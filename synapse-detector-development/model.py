"""Train a neural network model on synapse ground truth.

Currently, this code only trains Keras models, however it can be extended to
dynamically detect models from other frameworks and train appropriately.
"""
from .dataset import load
from .generators import augmenting_generator

import keras.backend as K

def train_model(model, batch_size=17, epochs=50):
    """Train a neural network model.

    Parameters
    ----------
    model
        The model to train.
    batch_size : int
        The size of each training batch.
    epochs : int
        The number of epochs to train for.

    Returns
    -------
    model
        The trained model.
    """
    return train_keras_model(model, batch_size=batch_size, epochs=epochs)

def train_keras_model(model, train_gen, batch_size=17, epochs=50):
    """Train a neural network model.

    Parameters
    ----------
    model : `keras.models.Model` or `keras.models.Sequential`
        The Keras model to train.
    batch_size : int
        The size of each training batch.
    epochs : int
        The number of epochs to train for.

    Returns
    -------
    model : `keras.models.Model` or `keras.models.Sequential`
        The trained Keras model.
    """
    class CB(Callback):
        def __init__(self, m, i, o):
            self.m = m
            self.i = i
            self.o = o

        def on_epoch_end(self, epoch, logs):
            print("saving ep{}.h5".format(epoch))
            f = h5py.File('ep{}.h5'.format(epoch), 'w')
            pred = self.m.predict(self.i)
            f.create_dataset('pred', data=pred)
            f.create_dataset('i', data=self.i)
            f.create_dataset('o', data=self.o)
            f.close()

    raw, gt, dists = load()
    config = model.get_config()
    input_shape = config['config']['layers'][0]['config']['batch_input_shape']
    output_shape =
    train_gen = augmenting_generator(raw, gt, dists, input_shape, output_shape,
                                     batch_size)
    i, o = next(train_gen)
    while o.mean() < 0.01:
        i, o = next(train_gen)

    model.fit_generator(train_gen, 1, epochs, verbose=2,
                        callbacks=[CB(model, i, o)])

    return model
