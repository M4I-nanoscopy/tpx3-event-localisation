import sys
import h5py
import numpy as np

w = h5py.File(sys.argv[-1], 'w')

freq_tot = np.zeros((512 * 512, 1024), dtype='uint32')

for idx, filename in enumerate(sys.argv[1:-1]):
    f = h5py.File(filename, 'r')

    freq_chunk = f['freq_tot'][()]
    freq_tot += freq_chunk

    f.close()

w.create_dataset('freq_tot', (512 * 512, 1024), dtype='uint32', data=freq_tot)

w.close()