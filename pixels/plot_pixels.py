import h5py
import matplotlib.pyplot as plt
import sys
import matplotlib
import numpy as np
from matplotlib import cm
import matplotlib.lines as mlines
from lib.constants import *


def plot_incident_electron(ax, incident):
    ax.plot(incident[1] / pixel_size, incident[0] / pixel_size, 's', color='red', label='Incident', markersize=10)


def plot_prediction(ax, val, color, label):
    ax.plot( val[1] / pixel_size, val[0] / pixel_size, 'o', color=color, label=label, markersize=10)


def plot_predictions(ax, predictions):
    colors = cm.rainbow(np.linspace(0, 1, len(predictions)))

    for idx, pred in enumerate(predictions):
        plot_prediction(ax, predictions[pred], colors[idx], pred)


def plot(fig, pixel, incident, predictions, edge):
    # Plot ToT
    ax = fig.add_subplot(211)

    # Converting to float32, as float16 cannot be displayed by imshow (https://github.com/matplotlib/matplotlib/issues/15432)
    tot = pixel[0, :].astype(np.float32)

    plt.imshow(tot, aspect='equal', interpolation='none', cmap='Greys_r', vmin=0, vmax=100, extent=[0, n_pixels, n_pixels, 0])

    # Plot edge
    if edge is not None:
        # [xmin,xmax], [ymin,ymax]
        y0 = 0
        y1 = 10
        x0 = (y0 - edge[1]) / edge[0]
        x1 = (y1 - edge[1]) / edge[0]
        l = mlines.Line2D([x0, x1], [y0, y1], color='green')
        ax.add_line(l)

    # Plot incident electrons
    if incident is not None:
        plot_incident_electron(ax, incident)

    # Plot predictions if they exist
    if predictions is not None:
        plot_predictions(ax, predictions)

    # Add legend
    if predictions is not None or incident is not None:
        plt.legend(bbox_to_anchor=(-1.2, 1), loc=2, numpoints=1)

    # Hide axis
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # Set title
    plt.title('Time over Threshold')
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Clock ticks (A.U)', rotation=270)

    # Plot ToA
    ax = fig.add_subplot(212)

    # Converting to float32, as float16 cannot be displayed by imshow (https://github.com/matplotlib/matplotlib/issues/15432)
    toa = pixel[1, :].astype(np.float32)
    # Plot edge

    if edge is not None:
        # [xmin,xmax], [ymin,ymax]
        y0 = 0
        y1 = 10
        x0 = (y0 - edge[1]) / edge[0]
        x1 = (y1 - edge[1]) / edge[0]
        l = mlines.Line2D([x0, x1], [y0, y1], color='green')
        ax.add_line(l)

    # Give pixels with NaN a different color
    matplotlib.cm.Greys_r.set_bad('lightblue', 1.)

    plt.imshow(toa, aspect='equal', interpolation='none', cmap=matplotlib.cm.Greys_r,
               vmin=0, vmax=10, extent=[0, n_pixels, n_pixels, 0])
    # Plot incident electrons
    if incident is not None:
        plot_incident_electron(ax, incident)

    # Plot predictions if they exist
    if predictions is not None:
        plot_predictions(ax, predictions)

    # Hide axis
    ax.xaxis.set_visible(False)
    ax.yaxis.set_visible(False)
    # Set title
    plt.title('Delta Time of Arrival')
    # Add colorbar
    cbar = plt.colorbar()
    cbar.set_label('Clock ticks (A.U)', rotation=270)


if __name__ == "__main__":
    filename = sys.argv[1]

    f = h5py.File(filename, 'r')

    pixel, predictions, incident, edge = None, None, None, None

    if 'clusters' in f:
        pixel = f['clusters'][int(sys.argv[2]), :, :]
    if 'incidents' in f:
        incident = f['incidents'][int(sys.argv[2]), :]
    if 'edges' in f:
        edge = f['edges'][[int(sys.argv[2])]]
    if 'predictions' in f:
        predictions = dict()
        for pred in f['predictions']:
            predictions[pred] = f['predictions'][pred][int(sys.argv[2])]

    fig = plt.figure()

    plot(fig, pixel, incident, predictions, edge)

    plt.show()
