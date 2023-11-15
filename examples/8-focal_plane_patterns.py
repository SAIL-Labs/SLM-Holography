#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 12:30:31 2023

@author: forrest
"""
from astropy import units as u
import numpy as np
import matplotlib.pyplot as plt
import os
import sys

sys.path.append('..')

from slm_super_pixel_holography import SLMSuperPixel



# The global variables for initializing the SLM

x_pixels = 400
y_pixels = 400
x_dim = 10 * u.mm
y_dim = 10 * u.mm
wavelength = 1.55 * u.micron
e_diam_pixels = 200
focal_length = 10 * u.mm
max_phase_shift = 2 * np.pi
only_in_e_diam = True
pix_per_super = 2
less_than_2pi = False

# The initializer function for the patterns
test_slm = SLMSuperPixel(x_pixels, 
                          y_pixels, 
                          x_dim, 
                          y_dim, 
                          wavelength, 
                          e_diam_pixels, 
                          focal_length, 
                          max_phase_shift, 
                          only_in_e_diam, 
                          pix_per_super, 
                          less_than_2pi)


# creates a pattern to apply to the image plane or focal plane, this example is a filled in circle

Y, X = np.mgrid[-200:200, -200:200]
R = np.sqrt(Y**2 + X**2)
focal_array = np.zeros((400,400))
focal_array[R < 50] = 1

PSF_pixscale_x = 5 * u.arcsecond
PSF_pixscale_y = 5 * u.arcsecond

total_ampl = 0.75 * np.sum(focal_array)
set_amplitude = True

test_slm.focal_plane_image(focal_array, PSF_pixscale_x, PSF_pixscale_y, total_ampl, set_amplitude = set_amplitude)

# converts the pattern to the SLM using the double pixel method and returns an array that is in radians
# if the pattern is only encoded in the entrance pupil, if add_padding is true the array is padded to be the size of the SLM
add_padding = False
# the amplitude and phase for the padding
#padding_phase = 0
#padding_ampl = 1

slm_final_array = test_slm.double_pixel_convert(add_padding)#, 
                                                #ampl_padding = padding_ampl, 
                                                #phase_padding = padding_phase)


plt.figure(1)
plt.title('SLM Pattern')
plt.imshow(slm_final_array, cmap='viridis')
plt.colorbar()
plt.show()

plt.figure(2)
plt.title('SLM Amplitude Array')
plt.imshow(test_slm.SLM_ampl, cmap='inferno')
plt.colorbar()
plt.show()

plt.figure(3)
plt.title('SLM Phase Array')
plt.imshow(test_slm.SLM_phase, cmap='plasma')
plt.colorbar()
plt.show()

plt.figure(4)
plt.title('Requested Image Plane')
plt.imshow(focal_array, cmap='plasma')
plt.colorbar()
plt.show()

Y, X = np.mgrid[-e_diam_pixels/2:e_diam_pixels/2, -e_diam_pixels/2:e_diam_pixels/2]
mask_R = np.sqrt(Y**2 + X**2)

circular_mask = np.zeros((e_diam_pixels, e_diam_pixels))
circular_mask[mask_R <= e_diam_pixels/2] = 1

padded_slm_array = np.pad(np.exp(1j * slm_final_array) * circular_mask, int(e_diam_pixels * 1))

fft_image_plane = np.fft.fftshift((np.fft.fft2(np.fft.fftshift(padded_slm_array))))

plt.figure(5)
plt.title('Actual Image Plane')
plt.imshow(np.abs(fft_image_plane)[e_diam_pixels:e_diam_pixels*2, e_diam_pixels:e_diam_pixels*2], cmap='plasma')
plt.colorbar()
plt.show()