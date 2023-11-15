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
x_pixels = 300
y_pixels = 200
x_dim = 10 * u.mm
y_dim = 20 * u.mm
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

# Sets the Zernike terms to use, using noll indices.  The scaling factor list needs to be of the same length as the noll indices list
# Noll indiices 1- flat phase, 2 - horizontal tilt, 3 - vertical tip, 4 - defocus, 
# 5 & 6 - (oblique, vertical) astigmatism, 7,8 - (vertical, horiztontal) coma, etc.

noll_inidices = [3, 5]
noll_scale    = [0.5, 0.5]

test_slm.zernike_terms(noll_inidices, noll_scale)


# converts the pattern to the SLM using the double pixel method and returns an array that is in radians
# if the pattern is only encoded in the entrance pupil, if add_padding is true the array is padded to be the size of the SLM
add_padding = True
# the amplitude and phase for the padding
padding_phase = 0
padding_ampl = 1

slm_final_array = test_slm.double_pixel_convert(add_padding, ampl_padding = padding_ampl, phase_padding = padding_phase)


plt.figure(1)
plt.title('SLM pattern')
plt.imshow(slm_final_array, cmap='viridis')
plt.colorbar()
plt.show()

plt.figure(2)
plt.title('SLM amplitude array')
plt.imshow(test_slm.SLM_ampl, cmap='inferno')
plt.colorbar()
plt.show()

plt.figure(3)
plt.title('SLM phase array')
plt.imshow(test_slm.SLM_phase, cmap='plasma')
plt.colorbar()
plt.show()