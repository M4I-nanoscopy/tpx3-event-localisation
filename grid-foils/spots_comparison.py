import sys
import numpy as np
from PIL import Image
from scipy import fftpack
import matplotlib.pyplot as plt
import spot_locations

plt.rcParams.update({
    "font.size": 12,
    "font.family": 'sans-serif',
    "svg.fonttype": 'none'
})

if sys.argv[1] == "300":
    spots = spot_locations.spots_300
else:
    spots = spot_locations.spots_200

frames = list()

for idx, filename in enumerate(sys.argv[2:4]):
    # Read image
    src_im = Image.open(filename)
    frame = np.array(src_im)

    if idx > 0:
        # Normalize counts in image to the square of the counts
        print(np.sum(frames[0]) / np.sum(frame))
        frame = frame * ( np.sum(frames[0] ) / np.sum(frame))

    frames.append(frame)


def img_spot_values(frame, s):
    # This zero pads the image to 1024x1024, this helps the FFT transform and is also done by ImageJ
    frame_padded = np.pad(frame, ((0, 1024 - frame.shape[0]), (0, 1024 - frame.shape[1])), 'constant')

    # Take the fourier transform of the image.
    F1 = fftpack.fft2(frame_padded)
    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    F2 = fftpack.fftshift(F1)
    # Calculate a 2D power spectrum
    ps = np.abs(F2) ** 2

    # Take the 10 log (this is normally only done for display purposes, not needed here)
    # ps = np.log10(psd2D)

    values = []

    for spot in s:
        #print ("x: %d, y: %d --> %d" % (spot[1], spot[0], ps[spot[0], spot[1]]))
        values.append(ps[spot[0], spot[1]])

    return np.array(values)


fig, axes = plt.subplots(nrows=1, figsize=(5,5))
ref = img_spot_values(frames[0], spots)
imp = img_spot_values(frames[1], spots)

improvement = imp / ref

for index, val in np.ndenumerate(improvement):
    print('{} => {}'.format(spots[index], val))

frequency = np.sqrt((spots[:, 0] - 512) ** 2 + (spots[:, 1] - 512) ** 2) / 512

axes.scatter(frequency, improvement)
axes.set_xlabel('Fraction of Nyquist')
axes.set_ylabel('MTF enhancement')

axes.grid()
axes.set_xlim((0,1))

if len(sys.argv) > 4:
    plt.savefig(sys.argv[4], dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()

