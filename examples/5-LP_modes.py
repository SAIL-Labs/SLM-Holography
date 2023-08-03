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


# The variables for the specific LP modes, 
# the diameter for the LP modes is defined by the values for the step index multimode fiber

# the requested fiber parameters
N_modes = 55
n_core = 1.44
n_cladding = 1.43


# lists of the requested modes L number, m number, intenisty to scale them by, and wether or not to make them odd
el = [0,1,3]
m = [1,1,1]
intensity_list = [0.2,0.5,0.3]
make_odd = [False, False, False]

# total amplitude scaling for the fourier transform
total_intensity = 500
set_intensity = False

#oversample and overscaling factors to increase the size and resolution of the requested lp modes
oversize = 5
oversample = 2

test_slm.LP_mode_encoding(N_modes, 
                          el, 
                          m, 
                          intensity_list, 
                          n_core, 
                          n_cladding, 
                          total_intensity, 
                          make_odd = make_odd, 
                          oversize = oversize, 
                          oversample = oversample, 
                          set_intensity = set_intensity)

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
plt.title('Requested LP modes')
plt.imshow(np.abs(test_slm.fourier_encode), cmap='plasma')
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