import argparse
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator

plt.rcParams.update({
    "font.size": 12,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})


def parse_arguments():
    parser = argparse.ArgumentParser(
        description=__doc__,  # printed with -h/--help
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--c200", metavar='FILE', help="200 kv uncorrected")
    parser.add_argument("--u200", metavar='FILE', help="200 kv corrected")
    parser.add_argument("--c300", metavar='FILE', help="300 kv uncorrected")
    parser.add_argument("--u300", metavar='FILE', help="300 kv corrected")
    parser.add_argument("--out", metavar='FILE', help="output")

    settings = parser.parse_args()

    return settings


def plot(f, ax, label, style, color):
    bins = np.arange(1, 600, 1)

    with h5py.File(f, 'r') as h5f:
        cluster_stats = h5f['cluster_stats'][()]
        tot_summed = cluster_stats[:, 1]

        y, edges = np.histogram(tot_summed, bins=bins, normed=True)
        centers = 0.5 * (edges[1:] + edges[:-1])

        hist = ax.plot(centers, y, label=label, linestyle=style, linewidth=2, color=color)

        # Find the fwhm
        hmx = half_max_x(centers, y)
        fwhm = hmx[1] - hmx[0]
        print("FWHM ({}): {:.3f}".format(label, fwhm))

    return hist


def lin_interp(x, y, i, half):
    return x[i] + (x[i + 1] - x[i]) * ((half - y[i]) / (y[i + 1] - y[i]))


def half_max_x(x, y):
    half = max(y) / 2.0
    signs = np.sign(np.add(y, -half))
    zero_crossings = (signs[0:-2] != signs[1:-1])
    zero_crossings_i = np.where(zero_crossings)[0]
    return [lin_interp(x, y, zero_crossings_i[0], half),
            lin_interp(x, y, zero_crossings_i[1], half)]


config = parse_arguments()

fig = plt.figure(dpi=200, figsize=(8, 4))
ax = fig.add_subplot(111)

if config.c200:
    plot(config.c200, ax, '200 kV corrected', 'solid', 'C0')

if config.u200:
    plot(config.u200, ax, '200 kV uncorrected', '--', 'C0')

if config.c300:
    plot(config.c300, ax, '300 kV corrected', 'solid', 'C1')

if config.u300:
    plot(config.u300, ax, '300 kV uncorrected', '--', 'C1')

plt.ylabel('Normalised occurrence')
plt.xlabel('Cluster ToT sum (A.U)')
plt.legend(loc='upper left')

ax.xaxis.set_major_locator(MultipleLocator(50))
plt.grid()

if config.out:
    plt.savefig(config.out, bbox_inches='tight')

plt.show()
