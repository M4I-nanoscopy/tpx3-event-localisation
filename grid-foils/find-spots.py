import sys
import numpy as np
from PIL import Image
from scipy import fftpack
import matplotlib.pyplot as plt
import matplotlib.colors as colors
import scipy.ndimage.filters as filters
import scipy.ndimage as ndimage

img_file = sys.argv[1]

# Read image
src_im = Image.open(img_file)
frame = np.array(src_im)

##############
# FFT of image
# uint8 max (255)
i8 = np.iinfo(np.uint8)

# This zero pads the image to 1024x1024, this helps the FFT transform and is also done by ImageJ
frame_padded = np.pad(frame, ((0, 1024-frame.shape[0]), (0, 1024-frame.shape[1])), 'constant')

# Take the fourier transform of the image.
F1 = fftpack.fft2(frame_padded)
# Now shift the quadrants around so that low spatial frequencies are in
# the center of the 2D fourier transformed image.
F2 = fftpack.fftshift(F1)
# Calculate a 2D power spectrum
psd2D = np.abs(F2) ** 2
# Take the 10 log
ps = np.log10(psd2D)

# Fit in uint8 (this is what ImageJ also does)
ps = i8.max * ps.astype(np.float64) / ps.max()

#####################
# Find suitable spots
neighborhood_size = 15
threshold = 80

data_max = filters.maximum_filter(ps, neighborhood_size)
maxima = (ps == data_max)
data_min = filters.minimum_filter(ps, neighborhood_size)
diff = ((data_max - data_min) > threshold)
maxima[diff == 0] = 0

labeled, num_objects = ndimage.label(maxima)
xy = np.array(ndimage.center_of_mass(ps, labeled, range(1, num_objects + 1)))

# Apply some filtering to get best spots
xy_filt = xy[np.bitwise_and(xy[:, 1] < 1000, xy[:, 1] > 550)]

# Take a random but uniform sample
choices = xy_filt[np.random.choice(len(xy_filt), size=50, replace=False)]
np.set_printoptions(linewidth=np.inf)

print("np.array([ ")
for choice in choices:
    print("    [%d, %d], " % (choice[0], choice[1]))
print("])")


fig, axes = plt.subplots(nrows=1)
axes.imshow(ps, vmin=ps.min(), vmax=ps.max())
axes.plot(xy_filt[:, 1], xy_filt[:, 0], 'ro')
axes.plot(choices[:, 1], choices[:, 0], 'go')


plt.show()
