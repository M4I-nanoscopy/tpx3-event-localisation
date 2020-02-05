import sys

import matplotlib
import matplotlib.pyplot as plt
import h5py
import os

from matplotlib import colors
from matplotlib import cm
import numpy as np
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from pixels import plot_pixels
from lib.constants import *

# File
filename = sys.argv[1]
f = h5py.File(filename, 'r')
clusters = f['clusters'][()]
incidents = f['incidents'][()]
predictions = f['predictions']

prediction_order = [
    'Random',
    'Centroid',
    'Highest ToA',
    'Highest ToT',
    'CNN-ToT',
    'CNN-ToT-ToA'
]

# Get indices
indices = list(map(int, sys.argv[2:5]))

# Figure
fig = plt.figure(figsize=(8, 4))
fig.subplots_adjust(left=0.1, hspace=0.1, top=0.83)

toa_cmap = matplotlib.cm.get_cmap('Greys_r')
toa_cmap.set_bad('lightblue', 1.)
color = cm.Set1(np.linspace(0, 1, 8))

for idx, i in enumerate(indices):
    ax = fig.add_subplot("24" + str(idx + 2))

    ax.set_title("Example " + str(idx + 1))

    # Plot all elements
    im_tot = ax.imshow(clusters[i, 0, 0:5, 0:5], aspect='equal', interpolation='none', cmap='Greys_r',
                      vmin=1, vmax=150, extent=[0, 5, 5, 0], norm=colors.LogNorm(vmin=1, vmax=150))
    plot_pixels.plot_incident_electron(ax, incidents[i])

    # Plot predictions
    for c, label in enumerate(prediction_order):
        pred = predictions[label][i]
        ax.plot(pred[1] / pixel_size, pred[0] / pixel_size, 'o', color=color[c], label=label, markersize=9)

    # Show grid
    ax.get_xaxis().set_ticks([0, 1, 2, 3, 4, 5])
    ax.get_yaxis().set_ticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, which='both', color='white')

    # Hide axis labels
    for tic in ax.xaxis.get_major_ticks():
        tic.tick2line.set_visible(False)
        tic.tick1line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick2line.set_visible(False)
        tic.tick1line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)

    # ToA ########
    ax = fig.add_subplot("24" + str(idx + 2 + 4))

    # Plot all elements for ToA graph
    im_toa = ax.imshow(clusters[i, 1, 0:5, 0:5], aspect='equal', interpolation='none', cmap=toa_cmap,
                       vmin=0, vmax=40, extent=[0, 5, 5, 0])
    plot_pixels.plot_incident_electron(ax, incidents[i])

    for c, label in enumerate(prediction_order):
        pred = predictions[label][i]
        ax.plot(pred[1] / pixel_size, pred[0] / pixel_size, 'o', color=color[c], label=label, markersize=10)

    # Show grid
    ax.get_xaxis().set_ticks([0, 1, 2, 3, 4, 5])
    ax.get_yaxis().set_ticks([0, 1, 2, 3, 4, 5])
    ax.grid(True, which='both', color='white')

    # Hide axis labels
    for tic in ax.xaxis.get_major_ticks():
        tic.tick2line.set_visible(False)
        tic.tick1line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)
    for tic in ax.yaxis.get_major_ticks():
        tic.tick2line.set_visible(False)
        tic.tick1line.set_visible(False)
        tic.label1.set_visible(False)
        tic.label2.set_visible(False)


# get left most plots
left_top = fig.axes[0]
left_bottom = fig.axes[1]

left_top.set_ylabel('ToT (A.U)', rotation=90, fontsize=12, labelpad=0)
left_bottom.set_ylabel(r"$\Delta$ToA (A.U)", rotation=90, fontsize=12, labelpad=0)

# Colorbars
# [x0, y0, width, height]
cax_tot = fig.add_axes([0.92, 0.49, 0.02, 0.34])
cax_toa = fig.add_axes([0.92, 0.10, 0.02, 0.34])

# noinspection PyUnboundLocalVariable
cbar_tot = fig.colorbar(im_tot, cax=cax_tot, orientation='vertical')
# cbar_tot.set_label('ToT (A.U)', rotation=270)
# noinspection PyUnboundLocalVariable
cbar_toa = fig.colorbar(im_toa, cax=cax_toa, orientation='vertical')
# cbar_toa.set_label(r"$\Delta$ToA (A.U)", rotation=270)

# Legend
ax_legend = fig.add_subplot(241)
ax_legend.axis("off")
handles, labels = ax.get_legend_handles_labels()
ax_legend.legend(ax, handles=handles, labels=labels)

if len(sys.argv) > 5:
    print(fig.dpi)
    plt.savefig(sys.argv[5], dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()
