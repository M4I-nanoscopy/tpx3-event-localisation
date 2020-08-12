import os
import sys
import h5py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.pyplot import cm
from scipy.optimize import curve_fit

plt.rcParams.update({
    #"font.size": 15,
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

    dist = np.linalg.norm(incidents - prediction, axis=1)

    return np.divide(dist, 55000)


distances = dict()

for pred in f['predictions']:
    distances[pred] = calculate_distance(f['predictions'][pred])

# Setup plots
fig = plt.figure(figsize=(10, 4))

ax = fig.add_subplot(321)
ax.set_ylabel('Distance (pixel)')
ax.yaxis.grid(True)
ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

# Build plots
boxes = []
for pred in prediction_order_complete:
    boxes.append(distances[pred])

    print("Median %s: %f" % (pred, np.median(distances[pred])))
    print("Mean %s: %f" % (pred, np.mean(distances[pred])))
    print("RMSD %s: %f" % (pred, np.sqrt(np.mean(np.square(distances[pred])))))

ax.boxplot(boxes, labels=prediction_order_complete, showfliers=False)

#method = 'CNN-ToT' # 4
method = 'CNN-ToT-ToA' # 5
ax = fig.add_subplot(322)
s = boxes[4]
print(prediction_order_complete[4])
edges = np.linspace(0.0, 4.0, 100)

H, edges = np.histogram(s, bins=edges)
bincenters = 0.5*(edges[1:]+edges[:-1])
H = H / H.sum()
ax.plot(bincenters, H, '-', color='green')
ax.set_xlim(0, 4)
ax.set_title(method)
ax.set_xlabel('Distance (pixel)')
ax.set_ylabel('Occurence')

def gauss_poly(x, *p):
    lam, = p
    return (2*x*lam**2)*np.exp(-(x**2)/(lam**2))

# p0 = [0.70,]
# coef, var_matrix = curve_fit(gauss_poly, edges[1:], H.transpose(), p0=p0)
# fit1 = gauss_poly(edges[1:], *p0)
# fit1 = fit1 / fit1.sum()
# print(coef)
# plt.plot(edges[1:], fit1, label="r exp(-(r)^2/(%.2f^2))" % (coef[0]))
# plt.legend()

# distributie = np.random.choice(edges[:-1], 10000000, p=fit1)
#
# plt.text(1.5, 0.02, "Median %s: %0.2f" % ("", np.median(distributie)))
# plt.text(1.5, 0.01, "Mean %s: %0.2f" % ("", np.mean(distributie)))
# plt.text(1.5, 0.0, "RMSD %s: %0.2f" % ("", np.sqrt(np.mean(np.square((distributie))))))


def calculate_distance_axis(prediction, axis):
    # Not sure this is the cleanest way, but it works to do sqrt( (x-x)^2 + (y - y)^2) on the whole matrix at once
    diff = incidents[:, axis] - prediction[:, axis]

    r = np.divide(diff, 55000)

    return r

ax = fig.add_subplot(323)

# Calculate 1 axis
distance_axis = calculate_distance_axis(f['predictions'][method], 1)

edges = np.arange(-10.0, 10.0, 0.1)

H, edges = np.histogram(distance_axis, bins=edges)
bincenters = 0.5*(edges[1:]+edges[:-1])
H = H / H.sum()
H_distance_axis = H
ax.plot(bincenters, H, '-', color='green')
ax.set_xlim(-4, 4)
ax.set_xlabel('x0-x')
ax.set_ylabel('Occurence')


def gauss(x, *p):
    A, mu, sigma = p
    return A*np.exp(-(x-mu)**2/(2.*sigma**2))

def psf(x, *p):
    lam, = p
    return 1/(np.pi*lam**2)*np.exp(-(x)**2/(lam**2))

def psf_2d_fit(coor, *p):
    X, Y = coor
    x0, y0 = 0, 0
    lam, = p
    r = 1/(np.pi*lam**2)*np.exp(-((X-x0)**2/(lam**2)+((Y+y0)**2/(lam**2))))

    # Outputs needs to be 1D. What you can do is add a .ravel() onto the end of the last line, like this:
    return r.ravel()


def psf_radial(r, *p):
    lam, r0 = p
    return 1/(np.pi*lam**2)*np.exp(-(r-r0)**2/(lam**2))


def psf_radial2(r, *p):
    lam, A, = p
    return A*np.exp(-(r)**2/(lam**2))

def mtf_gauss(lam):
    x = np.arange(0.01, 1.1, 0.01)

    y = np.exp(-np.pi**2*lam**2*x**2/4)

    return x, y

def theoretical_mtf():
    x = np.arange(0.01, 1.1, 0.01)

    y = np.sin(np.pi * x / 2) / (np.pi * x / 2)

    return x, y

def mtf_finite_pixel(lam):
    x, mtf = theoretical_mtf()

    y = mtf*np.exp(-np.pi**2*lam**2*x**2/4)

    return x, y

# Radial profile
def radial_profile(data, center):
    y, x = np.indices((data.shape))
    r = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    r = r.astype(np.int)

    tbin = np.bincount(r.ravel(), data.ravel())
    nr = np.bincount(r.ravel())
    radialprofile = tbin / nr
    return radialprofile

p0 = [1., 0., 1.]
coef, var_matrix = curve_fit(gauss, edges[1:], H, p0=p0)
fit1 = gauss(edges[1:], *coef)
print(coef)
plt.plot(edges[1:], fit1, label="%.2f exp(-(x-%.2f)^2/(%.2f^2))" % (coef[0], coef[1], coef[2]))
plt.legend()
plt.text(1.5, 0.02, "Median %s: %0.2f" % ("", np.median(distance_axis)))
plt.text(1.5, 0.01, "Mean %s: %0.2f" % ("", np.mean(distance_axis)))
plt.text(1.5, 0.0, "RMSD %s: %0.2f" % ("", np.sqrt(np.mean(np.square((distance_axis))))))

# 2D histogram of projected distance
fig.add_subplot(324)

distance_x = calculate_distance_axis(f['predictions'][method], 0)
distance_y = calculate_distance_axis(f['predictions'][method], 1)
H, _, _ = np.histogram2d(distance_x, distance_y, bins=[edges, edges])
#H = H / H.sum()
plt.imshow(H, origin='low',extent=[edges[0], edges[-1], edges[0], edges[-1]])
plt.xlim(-2, 2)
plt.ylim(-2, 2)

# Radial fit
fig.add_subplot(325)

# Calculate radial profile of normalized plot
rad_prof = radial_profile(H, [100, 100])
sum_psf = np.concatenate((np.flip(rad_prof), rad_prof))
sum_psf_norm = sum_psf / sum_psf.max()
s = np.sqrt(10**2+10**2)
r = np.arange(-s, s+0.1, 0.1)
plt.plot(r, sum_psf_norm, label='Radial PSF')
plt.xlim(-2, 2)
p0 = [.50, 1]
coef, var_matrix = curve_fit(psf_radial2, r, sum_psf_norm, p0=p0)
print(coef)
fit = psf_radial2(r, *p0)
lab = 0.50
test = psf(r, *[lab])
plt.plot(r, test/test.max(), label='PSFg (exp(-x^2/(%.2f^2)/pi*%.2f^2)' % (lab, lab))
plt.plot(r, fit, label="%.2f exp(-(x)^2/(%.2f^2))" % (coef[1], coef[0]))
plt.legend()
plt.title("Radial plot of 2D plot")

# MTF
fig.add_subplot(326)
mtf = np.abs(np.fft.fft(sum_psf_norm))

mtf_norm = mtf/mtf[0]
plt.title("MTF")
plt.plot(mtf_norm, label="FFT of fit")
x, y = mtf_gauss(lab)
plt.plot(x*s, y, label="Analytical MTFg (lambda=%0.2f)" % lab)
x, y = mtf_finite_pixel(lab)
plt.plot(x*s, y, label="MTF finite pixel (lambda=%0.2f)" % lab)
x, y = theoretical_mtf()
plt.plot(x*s, y, label="Theoretical MTF")
plt.xlim(0, s)
plt.ylim(0, 1)
plt.legend()
if len(sys.argv) > 2:
    plt.savefig(sys.argv[2], dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.subplots_adjust(hspace=0.50)
plt.show()
