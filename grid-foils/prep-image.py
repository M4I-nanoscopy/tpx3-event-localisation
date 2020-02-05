import PIL
import sys
import numpy as np
import os
from PIL import Image, ImageFont
from PIL import ImageDraw
from scipy import fftpack
import spot_locations

if sys.argv[1] == "300":
    spots = spot_locations.spots_300
else:
    spots = spot_locations.spots_200

img_file = sys.argv[2]
base = os.path.basename(img_file)
img_file_no_ext = os.path.splitext(base)[0]

# uint8 max (255)
i8 = np.iinfo(np.uint8)

# Read image
src_im = Image.open(img_file)
frame = np.array(src_im)

##################
# Convert to uint8
if frame.max() > i8.max:
    print("WARNING: Cannot fit in uint8. Scaling values to uint8 max.")
    data = i8.max * frame.astype(np.float64) / frame.max()
    frame_uint8 = data.astype(np.uint8)
else:
    frame_uint8 = frame.astype(dtype=np.uint8)

# Save as uint8 image
im_real = Image.fromarray(frame_uint8)
# im.save(img_file_no_ext + '-uint8.tif')

##############
# FFT of image

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

# Invert the image. To make
ps = i8.max - ps

# Threshold FFT
min1 = np.percentile(ps, 1)
max1 = np.percentile(ps, 9)
print("Min: %d" % min1)
print("Max: %d" % max1)
ps = np.clip(ps, min1, max1)

# Rescale values
ps = np.interp(ps, (min1, max1), (0, i8.max))

im_fft = Image.fromarray(ps.astype(dtype=np.uint8))

#################
# Put all images together
margin = 2
half = 258
full = 516

# This RESIZES (or bins) the FFT to make it APPEAR to be the same size as the image
im_fft = im_fft.resize((516, 516), PIL.Image.NEAREST)
#im_fft.save(img_file_no_ext + '-fft.tif')

# Create white image
new_im = Image.new('L', (full + margin, full))
draw = ImageDraw.Draw(new_im)
draw.rectangle([(0, 0), new_im.size], fill=255)

# Take insert
insert = im_fft.resize((258, 256), PIL.Image.NEAREST, (half + 130, half + 130, half + 130 + 64, half + 130 + 64))

# Draw spots
draw = ImageDraw.Draw(im_fft)
# Recalculate spot positions after resizing
spots = spots * 516/1024
for spot in spots:
    draw.ellipse((spot[1] - 5, spot[0] - 5, spot[1] + 5, spot[0] + 5), outline=0)

# Draw insert
draw = ImageDraw.Draw(im_fft)
draw.rectangle([(130 + half, 130 + half), (130 + half + 64, 130 + half + 64)], outline=(0))

# Crops
im_fft_crop = im_fft.crop((half, 0, full, full))
im_real_crop = im_real.crop((0, 0, half, half))

# Combine
new_im.paste(im_fft_crop, (half + margin, 0, full + margin, full))
new_im.paste(im_real_crop, (0, 0, half, half))
new_im.paste(insert, (0, half + margin, half, full))

# Text
fnt = ImageFont.truetype('Pillow/Tests/fonts/FreeMono.ttf', 20)
d = ImageDraw.Draw(new_im)
#d.text((5, 5), "real", font=fnt, fill=128)
d.text((half + 5, 5), "fft", font=fnt, fill=0)
d.text((5, half + 5), "zoom", font=fnt, fill=0)

new_im.save(img_file_no_ext + '-prepped.tif')
# new_im.show()
