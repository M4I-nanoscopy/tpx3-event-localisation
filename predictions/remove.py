import sys
import h5py

filename = sys.argv[1]

with h5py.File(filename, 'a') as f:
    del f['predictions']
