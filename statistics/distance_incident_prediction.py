import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import cm

plt.rcParams.update({
    "font.size": 15,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})

prediction_order = [
    'Random',
    'Centroid',
    'Highest ToA',
    'Highest ToT',
    'CNN-ToT',
    'CNN-ToT-ToA'
]

filename = sys.argv[1]
f = h5py.File(filename, 'r')

incidents = f['incidents'][()]

sensor_height = f.attrs.get("sensor_height", "N/A")
beam_energy = f.attrs.get("beam_energy", "N/A"),
sensor_material = f.attrs.get("sensor_material", "N/A")

# Merge prediction order
prediction_order_complete = prediction_order
prediction_order_complete.extend(x for x in list(f['predictions']) if x not in prediction_order_complete)


def calculate_distance(prediction):
    # Not sure this is the cleanest way, but it works to do sqrt( (x-x)^2 + (y - y)^2) on the whole matrix at once
    diff = incidents - prediction
    square = np.square(diff)
    dist = np.sqrt(square[:, 0] + square[:, 1])

    return np.divide(dist, 55000)


distances = dict()

for pred in f['predictions']:
    distances[pred] = calculate_distance(f['predictions'][pred])

# Setup plots
fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(111)
ax.set_ylabel('Distance (pixel)')
ax.yaxis.grid(True)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# Build plots
boxes = []
for pred in prediction_order_complete:
    boxes.append(distances[pred])

    print("%s: %f" % (pred, np.median(distances[pred])))

ax.boxplot(boxes, labels=prediction_order_complete, showfliers=False)

if len(sys.argv) > 2:
    plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()
