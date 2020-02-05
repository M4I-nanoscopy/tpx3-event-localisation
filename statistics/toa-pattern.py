import sys
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
from scipy import fftpack


def imshow(frame, a, title):
    # Calculate events at super resolution
    min5 = np.percentile(frame, 1)
    max95 = np.percentile(frame, 99)

    a.imshow(frame, vmin=min5, vmax=max95, cmap='gray')

    # Look and feel
    a.xaxis.set_major_locator(plt.FixedLocator([len(frame[0]) - 1]))
    a.yaxis.set_major_locator(plt.FixedLocator([len(frame[0]) - 1]))
    a.set_title(title)

    return frame


def psshow(f, a, title):
    # Take the fourier transform of the image.
    f1 = fftpack.fft2(f)
    print(f.shape)

    # Now shift the quadrants around so that low spatial frequencies are in
    # the center of the 2D fourier transformed image.
    f2 = fftpack.fftshift(f1)

    # Calculate a 2D power spectrum
    psd2D = np.abs(f2) ** 2

    # Display at log 10
    psd2D_log = np.log10(psd2D)

    min5 = np.percentile(psd2D_log, 5)
    print(min5)
    max95 = np.max(psd2D_log)
    print(max95)

    a.imshow(psd2D_log, vmin=min5, vmax=max95, cmap='gray')

    # Look and feel
    a.xaxis.set_major_locator(plt.FixedLocator([len(psd2D[0]) - 1]))
    a.yaxis.set_major_locator(plt.FixedLocator([len(psd2D[0]) - 1]))
    a.set_title(title)


x = 10
y = 10
width = 128
height = 128

# Figure
fig = plt.figure()
plt.subplots_adjust(wspace=0.0)
# CNN-ToT
filename = sys.argv[1]
src_im = Image.open(filename)
im = np.array(src_im)

ax = fig.add_subplot("221")
imshow(im[y:y+height, x:x+width], ax, "CNN-ToT")

ax = fig.add_subplot("222")
psshow(im[y:y+height, x:x+width], ax, "CNN-ToT FFT")

# CNN-ToA
filename = sys.argv[2]
src_im = Image.open(filename)
im = np.array(src_im)

ax = fig.add_subplot("223")
imshow(im[y:y+height, x:x+width], ax, "CNN-ToT-ToA")

ax = fig.add_subplot("224")
psshow(im[y:y+height, x:x+width], ax, "CNN-ToT-ToA FFT")


# Save to file
if len(sys.argv) > 3:
    plt.savefig(sys.argv[3], dpi=300, bbox_inches='tight', pad_inches=0.1)

plt.show()
