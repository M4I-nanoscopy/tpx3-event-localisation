import sys
import os
import h5py
import numpy as np
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), os.path.pardir)))

plt.rcParams.update({
    "font.size": 12,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})

fig = plt.figure(dpi=200)
ax = fig.add_subplot(111)

f = h5py.File(sys.argv[1], 'r')

pixels = f['clusters'][:, 0, :]

size = list()
tot = list()
for pixel in pixels:
    tot.append(np.sum(pixel))
    size.append(np.count_nonzero(pixel))

# max_tot = np.percentile(tot, 99.99)
max_tot = int(sys.argv[2])
# max_size = np.percentile(size, 99.999)
max_size = int(sys.argv[3])

cmap = plt.get_cmap('viridis')
cmap.set_under('w', 1)
bins = [np.arange(0, max_tot, 25), np.arange(0, max_size, 1)]
plt.hist2d(tot, size, cmap=cmap, vmin=0.000001, range=((0, max_tot), (0, max_size)), bins=bins, normed=True)

# x-axis ticks
xax = ax.get_xaxis()
xax.set_major_locator(plt.MultipleLocator(50))
xax.set_minor_locator(plt.MultipleLocator(25))
xax.set_tick_params(colors='black', which='major')
plt.xlabel('ToT Sum (A.U)')

# y-axis ticks
yax = ax.get_yaxis()
yax.set_major_locator(plt.MultipleLocator(1))
yax.set_tick_params(colors='black', which='major')

ax.set_ylim(1)
plt.ylabel('Cluster Size (pixels)')

# Set grid
plt.grid(b=True, which='both')

# Colorbar
cbar = plt.colorbar()
cbar.set_ticks([])
cbar.set_label('Normalised occurrence')

if len(sys.argv) > 4:
    plt.savefig(sys.argv[4], bbox_inches='tight')

plt.show()
