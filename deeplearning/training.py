import argparse
import os
import sys

import keras
from keras.layers import Dense, Dropout, Flatten, SeparableConv2D
from keras.models import Sequential, load_model
from keras.optimizers import Adam
import h5py
import numpy as np
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lib.constants import *



def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('FILE', help="Input .h5 dataset")
    parser.add_argument("-e", "--epochs", default=200, metavar='N', type=int, help="Number of epochs")
    parser.add_argument("-b", "--batch_size", default=1000, metavar='N', help="Batch size for training")
    parser.add_argument("--model", metavar='FILE', help="Path to model")
    parser.add_argument("--tot", default=False, action='store_true', help="Train on only ToT")
    parser.add_argument("--toa", default=False, action='store_true', help="Train on only ToA")
    parser.add_argument("--append_model", default=False, action='store_true', help="Continue training model")
    parser.add_argument("--dropout", default=0.25, help="Dropout")
    parser.add_argument("--dense_size", default=512, help="Dense size")
    parser.add_argument("--train_rate", default=0.7, metavar='N', help="Percentage of training/test split")
    parser.add_argument("--classic", default=False, action='store_true', help='set to true for using classic NN')
    parser.add_argument("--minimal_cnn", default=False, action='store_true', help='set to true to use minimal CNN')
    settings = parser.parse_args()

    return settings


def new_model(dropout, dense_size, tot, toa, path, loss="logcosh"):

    if tot:
        shape = 1
    elif toa:
        shape = 1
    else:
        shape = 2

    model = Sequential()
    model.add(SeparableConv2D(64, (2, 2), padding='same', input_shape=(shape, n_pixels, n_pixels),
                              data_format="channels_first", activation="relu"))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(dense_size, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(dropout))
    model.add(Dense(dense_size, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(dropout))
    model.add(Dense(dense_size, activation="relu", kernel_initializer="uniform"))
    model.add(Dropout(dropout))
    model.add(Dense(2, activation="relu"))
    adam = Adam(lr=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy', 'mse'])
    model.save(path)


def minimal_cnn_model(dropout, dense_size, tot, toa, path, loss="logcosh"):
    if tot:
        shape = 1
    elif toa:
        shape = 1
    else:
        shape = 2

    model = Sequential()
    model.add(SeparableConv2D(64, (2, 2), padding='same', input_shape=(shape, n_pixels, n_pixels),
                              data_format="channels_first", activation="relu"))
    model.add(Dropout(dropout))
    model.add(Flatten())
    model.add(Dense(2, activation="relu"))

    adam = Adam(lr=0.001)
    model.compile(loss=loss, optimizer=adam, metrics=['accuracy', 'mse'])
    model.save(path)


def classic_model(dropout, dense_size, tot, toa, path, loss="logcosh"):
    model = Sequential()
    if tot:
        shape = 1
    elif toa:
        shape = 1
    else:
        shape = 2

    model.add(Dense(256, activation="relu", input_shape=(shape, n_pixels, n_pixels)))
    model.add(Flatten())
    model.add(Dense(128, activation="relu"))
    model.add(Dense(64, activation="relu"))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(16, activation="relu"))
    model.add(Dense(8, activation="relu"))
    model.add(Dense(4, activation="relu"))
    model.add(Dense(2, activation="relu"))
    adam = Adam(lr=0.001)

    model.compile(loss=loss, optimizer=adam, metrics=['accuracy', 'mse'])
    model.save(path)


def train_data(pixels, coords, tot, toa, train_rate):
    if tot:
        shape = 1
    elif toa:
        shape = 1
    else:
        shape = 2

    # Replacing the nans
    pixels = np.nan_to_num(pixels)

    n = len(pixels)
    indexList = range(n)
    train_indices = np.random.choice(n, size=int(float(train_rate) * n), replace=False)
    test_indices = list(set(indexList).difference(train_indices))

    x_train, y_train = np.zeros((len(train_indices), shape, n_pixels, n_pixels)), np.zeros((len(train_indices), 2))
    x_test, y_test = np.zeros((len(test_indices), shape, n_pixels, n_pixels)), np.zeros((len(test_indices), 2))

    if tot:
        x_train[:, 0] = pixels[train_indices, 0]
    elif toa:
        x_train[:, 0] = pixels[train_indices, 1]
    else:
        x_train[:, 0] = pixels[train_indices, 0]
        x_train[:, 1] = pixels[train_indices, 1]

    y_train[:, 0] = coords[train_indices, 0] / pixel_size
    y_train[:, 1] = coords[train_indices, 1] / pixel_size

    if tot:
        x_test[:, 0] = pixels[test_indices, 0]
    elif toa:
        x_test[:, 0] = pixels[test_indices, 1]
    else:
        x_test[:, 0] = pixels[test_indices, 0]
        x_test[:, 1] = pixels[test_indices, 1]

    y_test[:, 0] = coords[test_indices, 0] / pixel_size
    y_test[:, 1] = coords[test_indices, 1] / pixel_size

    return x_train, y_train, x_test, y_test


def train(training_path, epochs, batch_size, tot, toa, model_path, train_rate):
    model = load_model(model_path)
    f = h5py.File(training_path, "r")
    pix = f['clusters'][()]
    coords = f['incidents'][()]

    x_train, y_train, x_test, y_test = train_data(pix, coords, tot, toa, train_rate)
    name = os.path.splitext(os.path.basename(model_path))[0]

    history = keras.callbacks.TensorBoard(
        log_dir="./tb/%s" % name, histogram_freq=25, batch_size=32, write_graph=True, write_grads=False,
        write_images=False,
        embeddings_freq=0, embeddings_layer_names=None, embeddings_metadata=None
    )

    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs,
              validation_data=(x_test, y_test),
              shuffle=True,
              # callbacks=[history]
              )

    score = model.evaluate(x_test, y_test, batch_size=200)

    model.save(model_path)
    model.summary()


def main():
    config = parse_arguments()

    if not config.append_model:
        if config.classic:
            classic_model(config.dropout, config.dense_size, config.tot, config.toa, config.model)
        elif config.minimal_cnn:
            minimal_cnn_model(config.dropout, config.dense_size, config.tot, config.toa, config.model)
        else:
            new_model(config.dropout, config.dense_size, config.tot, config.toa, config.model)

    train(config.FILE, config.epochs, config.batch_size, config.tot, config.toa, config.model, config.train_rate)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
