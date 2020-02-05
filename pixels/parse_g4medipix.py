import numpy as np
import h5py
import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))
from lib.constants import *

filename = sys.argv[1]

f = h5py.File(filename, 'a')

used_n_pixels = f.attrs['n_pixels']
sensor_height = f.attrs['sensor_height']

if 'incidents' in f:
    del f['incidents']

f.create_dataset("incidents", (len(f['g4medipix']), 2))

pixels = f['g4medipix']

if 'clusters' in f:
    del f['clusters']

f.create_dataset("clusters", (len(f['g4medipix']), 2, n_pixels, n_pixels))

for idx in pixels:
    # Read dataset
    pixel = pixels[idx][()]

    tot = pixel[0, :]
    toa = pixel[1, :]

    if np.count_nonzero(tot) == 0:
        continue

    # Calculate the shift to be taken from the ToT values
    y, x = np.nonzero(tot)
    shift_x = min(x)
    shift_y = min(y)

    # Apply shift to tot and toa matrices
    tot = np.roll(tot, -shift_x, axis=1)
    tot = np.roll(tot, -shift_y, axis=0)
    toa = np.roll(toa, -shift_x, axis=1)
    toa = np.roll(toa, -shift_y, axis=0)

    # G4Medipix outputs a few zero values around clusters, convert those to -NaN
    toa[toa == 0] = -np.nan

    # Reduce the lowest ToA values to 0, as we do not know a time offset. Just like the real data
    toa = toa - np.nanmin(toa)

    # Store values  in resized matrix again
    f['clusters'][int(idx), 0, :] = tot[0:n_pixels, 0:n_pixels]
    f['clusters'][int(idx), 1, :] = toa[0:n_pixels, 0:n_pixels]

    trajectory = f['trajectories'][str(idx)][()]

    # G4medipix sets its origin in the middle of four pixels
    trajectory[:, 1] = trajectory[:, 1] + (-1 + used_n_pixels / 2 - shift_x) * pixel_size
    trajectory[:, 0] = trajectory[:, 0] + (-1 + used_n_pixels / 2 - shift_y) * pixel_size
    # Invert trajectory, and increase with half sensor height
    trajectory[:, 2] = trajectory[:, 2] * -1 + sensor_height / 2

    # Store trajectory back
    del f['trajectories'][str(idx)]
    # Sort by time
    f['trajectories'][str(idx)] = trajectory[trajectory[:, 4].argsort()]

    # Calculate incident position
    f['incidents'][int(idx)] = [trajectory[0, 0], trajectory[0, 1]]

# Store new cluster size
f.attrs['n_pixels'] = n_pixels

del f['g4medipix']
