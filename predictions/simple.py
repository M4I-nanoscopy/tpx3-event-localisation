import random
import h5py
from scipy import ndimage
import numpy as np
import os
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lib.constants import *

filename = sys.argv[1]
f = h5py.File(filename, 'a')

highest_tot = "Highest ToT"
highest_toa = "Highest ToA"
weighted_centroid = "Centroid"
# centroid = "Centroid"
rand = "Random"

# Raise runtime warnings, instead of printing them
np.seterr(all='raise')

# Get just the ToT pixels
plots = f['clusters'][:, 0, :]

if 'predictions' in f and highest_tot in f['predictions']:
    del f['predictions'][highest_tot]

num_plots = plots.shape[0]
results = f.create_dataset("/predictions/%s" % highest_tot, (num_plots, 2))

# Very naive Charge Sharing Mode (ToT)
for idx, pixels in enumerate(plots):
    i = pixels.argmax()
    y, x = np.unravel_index(i, (n_pixels, n_pixels))

    results[idx] = [(y + 0.5) * pixel_size, (x + 0.5) * pixel_size]

# Highest ToA

plots_toa = f['clusters'][:, 1, :]

if 'predictions' in f and highest_toa in f['predictions']:
    del f['predictions'][highest_toa]

results = f.create_dataset("/predictions/%s" % highest_toa, (num_plots, 2))

# Calculate highest_toa
for idx, pixels in enumerate(plots_toa):

    if np.nanmax(pixels) == 0:
        # Get the ToT data, and find a pixel in there
        nzy, nzx = np.nonzero(plots[idx])

        if len(nzy) == 0:
            continue
        elif len(nzy) == 1:
            y, x = nzy[0], nzx[0]
        else:
            i = np.random.randint(0, len(nzy) - 1)
            y, x = nzy[i], nzx[i]
    else:
        maxes = np.argwhere(pixels == np.nanmax(pixels))

        if len(maxes) > 1:
            i = np.random.randint(0, len(maxes) - 1)
            y, x = maxes[i][0], maxes[i][1]
        else:
            y, x = maxes[0][0], maxes[0][1]

    results[idx] = [(y + 0.5) * pixel_size, (x + 0.5) * pixel_size]

# Weighted Centroid

if 'predictions' in f and weighted_centroid in f['predictions']:
    del f['predictions'][weighted_centroid]

results = f.create_dataset("/predictions/%s" % weighted_centroid, (num_plots, 2))

for idx, pixels in enumerate(plots):

    try:
        y, x = ndimage.measurements.center_of_mass(pixels)
        results[idx] = [(y + 0.5) * pixel_size, (x + 0.5) * pixel_size]
    except FloatingPointError:
        print("Could not calculate center of mass: empty cluster?. Cluster_idx: %s" % idx)

# Centroid

# if 'predictions' in f and centroid in f['predictions']:
#     del f['predictions'][centroid]
#
# results = f.create_dataset("/predictions/%s" % centroid, (num_plots, 2))
#
# for idx, pixels in enumerate(plots):
#
#     # Normalize all values to 1
#     pixels[pixels > 0] = 1
#
#     try:
#         y, x = ndimage.measurements.center_of_mass(pixels)
#         results[idx] = [(y + 0.5) * pixel_size, (x + 0.5) * pixel_size]
#     except FloatingPointError:
#         print("Could not calculate center of mass: empty cluster?. Cluster_idx: %s" % idx)


# Random

if 'predictions' in f and rand in f['predictions']:
    del f['predictions'][rand]

results = f.create_dataset("/predictions/%s" % rand, (num_plots, 2))

for idx, pixels in enumerate(plots):

    nzy, nzx = np.nonzero(pixels)

    if len(nzy) == 0:
        print("Could not find random pixel: empty cluster?. Cluster_idx: %s" % idx)
        continue
    elif len(nzy) == 1:
        y, x = nzy[0], nzx[0]
    else:
        i = np.random.randint(0, len(nzy) - 1)
        y, x = nzy[i], nzx[i]

    results[idx] = [(y + 0.5 + random.uniform(-0.5, 0.5)) * pixel_size,
                    (x + 0.5 + random.uniform(-0.5, 0.5)) * pixel_size]


f.flush()
f.close()
