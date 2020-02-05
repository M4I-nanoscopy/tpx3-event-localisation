import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    "font.size": 12,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})

filename = sys.argv[1]
tot = int(sys.argv[2])

f = h5py.File(filename, 'r')

data = f['tot_correction']

print(data.attrs['creation_date'])

d = data[tot, :]

full = np.zeros((512,512))

full[0:256, 0:256] = np.fliplr(d[:, :, 2])
full[256:512, 0:256] = np.flipud(d[:, :, 3])
full[256:512, 256:512] = np.flipud(d[:, :, 0])
full[0:256, 256:512] = np.flipud(d[:, :, 1])

fig = plt.figure()
im = plt.imshow(full, vmin=-50, vmax=70)

ax = plt.gca()
ax.xaxis.set_major_locator(plt.FixedLocator([512 - 1]))
ax.yaxis.set_major_locator(plt.FixedLocator([512 - 1]))

fig.colorbar(im, label='ToT correction value')

if len(sys.argv) > 3:
    plt.savefig(sys.argv[3], bbox_inches='tight', pad_inches=0.1)


plt.show()