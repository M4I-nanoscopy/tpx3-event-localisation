import argparse
import os
import sys

import h5py
import numpy as np
from keras.models import load_model

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lib.constants import *


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument('FILE', help="Input .h5 dataset")
    parser.add_argument("--model", metavar='FILE', help="Path to model")
    parser.add_argument("--tot", default=False, action='store_true', help="Predict on only ToT")
    parser.add_argument("--toa", default=False, action='store_true', help="Predict on only ToA")
    parser.add_argument("--name", default='CNN', help="Name of prediction to use")
    parser.add_argument("--experimental", default=False, action='store_true',
                        help='set to true to use experimental data')

    settings = parser.parse_args()

    return settings


def predict(predictpath, model_path, tot, toa, predic):
    f = h5py.File(predictpath, "a")
    pixels = f['clusters'][()]
    n = len(pixels)

    if tot:
        shape = 1
    elif toa:
        shape = 1
    else:
        shape = 2

    x_test, y_test = np.zeros((n, shape, n_pixels, n_pixels)), np.zeros((n, 2))

    for i in range(0, n):
        if not toa and not tot:
            x_test[i, 0] = pixels[i, 0][0:n_pixels, 0:n_pixels]
            x_test[i, 1] = np.nan_to_num(pixels[i, 1])[0:n_pixels, 0:n_pixels]
        elif toa:
            x_test[i, 0] = np.nan_to_num(pixels[i, 1])[0:n_pixels, 0:n_pixels]
        elif tot:
            x_test[i, 0] = pixels[i, 0][0:n_pixels, 0:n_pixels]

    model = load_model(model_path)

    # keras.utils.plot_model(model, to_file='test.ps', show_shapes=True, rankdir='TB')
    # exit(0)

    pred = model.predict(x_test, batch_size=n, verbose=1)

    if not f.__contains__("predictions"):
        predictions = f.create_group("predictions")
    else:
        predictions = f["predictions"]

    pred = pred * 55000

    if predictions.__contains__(predic):
        del predictions[predic]

    predictions.create_dataset(predic, data=pred)


def predict3(predictpath, model_path, tot, toa, predic):
    f = h5py.File(predictpath, "a")
    pixels = f['clusters'][()]
    edges = f['edges'][()]
    n = len(pixels)

    if tot:
        shape = 1
    elif toa:
        shape = 1
    else:
        shape = 2

    x_test, y_test = np.zeros((n, shape, n_pixels, n_pixels)), np.zeros((n, 2))

    for i in range(0, n):
        if not toa and not tot:
            x_test[i, 0] = pixels[i, 0][0:n_pixels, 0:n_pixels]
            x_test[i, 1] = np.nan_to_num(pixels[i, 1])[0:n_pixels, 0:n_pixels]
        elif toa:
            x_test[i, 0] = np.nan_to_num(pixels[i, 1])[0:n_pixels, 0:n_pixels]
        elif tot:
            x_test[i, 0] = pixels[i, 0][0:n_pixels, 0:n_pixels]

    model = load_model(model_path, custom_objects={"loss_truth_mask": loss_truth_mask})

    pred = model.predict(x_test, batch_size=1000, verbose=1)

    if not f.__contains__("predictions"):
        predictions = f.create_group("predictions")

    else:
        predictions = f["predictions"]

    pred = pred * 55000

    if predictions.__contains__(predic):
        del predictions[predic]

    predictions.create_dataset(predic, data=pred)


def main():
    config = parse_arguments()

    if config.experimental:
        predict3(config.FILE, config.model, config.tot, config.toa, config.name)
    else:
        predict(config.FILE, config.model, config.tot, config.toa, config.name)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        sys.exit(0)
