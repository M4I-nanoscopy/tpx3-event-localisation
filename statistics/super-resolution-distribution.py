import sys
import h5py
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
from mpl_toolkits.axes_grid1 import make_axes_locatable
from mpl_toolkits.axes_grid1.inset_locator import inset_axes

super_resolution = 2
shape = 516 * super_resolution

plt.rcParams.update({
    "font.size": 12,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})

def colorbar(mappable):
    ax = mappable.axes
    fig = ax.figure
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    return fig.colorbar(mappable, cax=cax)


def imshow(e, a):
    # Calculate events at super resolution
    x = e['x'] * super_resolution
    y = e['y'] * super_resolution
    data = np.ones(len(x))
    d = scipy.sparse.coo_matrix((data, (y, x)), shape=(shape, shape), dtype=np.uint32)
    frame = d.todense()

    min5 = np.percentile(frame, 5)
    max95 = np.percentile(frame, 95)

    im = a.imshow(frame, vmin=min5, vmax=max95, cmap='gray')

    # Look and feel
    a.xaxis.set_major_locator(plt.FixedLocator([shape - 1]))
    a.yaxis.set_major_locator(plt.FixedLocator([shape - 1]))

    # Color bar
    # colorbar(im)

    return im


def hist2d(e, a):
    bins = np.arange(0, 1.1, 0.5)
    remain_x = np.mod(e['x'], np.ones(len(e)))
    remain_y = np.mod(e['y'], np.ones(len(e)))
    hist, _, _, im = a.hist2d(remain_x, remain_y, bins=[bins, bins], cmin=0, normed=True)
    im.set_clim(vmin=0)

    # Look and feel
    a.set_aspect('equal', adjustable='box')
    a.set_axis_off()

    # Plot values
    for i in range(len(bins) - 1):
        for j in range(len(bins) - 1):
            a.text(bins[j] + 0.25, bins[i] + 0.25, ".2%f" % hist[i, j], color="black", ha="center", va="center", fontweight="bold")


# Figure
fig = plt.figure(figsize=(10, 8))

# Without correction
filename = sys.argv[1]
f = h5py.File(filename, 'r')

events = f['events'][()]
ax = fig.add_subplot("111")
imshow(events, ax)

# Plot 2d histogram on an inset
ax_inset = inset_axes(ax, width="30%", height="30%", loc=4, borderpad=2)
hist2d(events, ax_inset)

if len(sys.argv) > 2:
    plt.savefig(sys.argv[2], bbox_inches='tight', pad_inches=0.1, pil_kwargs={'compression': 'tiff_deflate'})

plt.show()
