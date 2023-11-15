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

# sets the parameters of the gaussian, a is the max amplitude, b_x and b_y is the centering, and c_x and c_y is the standard deviation

ampl = 1
b_x = 100
b_y = 100
c_x = 1000
c_y = 1000

test_slm.gaussian_ampl(a = ampl, b_x = b_x, b_y = b_y, c_x = c_x, c_y = c_y)

# sets the phase to a given value between 0 and max_phase_shift
flat_phase_value = np.pi
test_slm.flat_phase(flat_phase_value)

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