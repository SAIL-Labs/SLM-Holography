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

from SLM_double_pixel_holography import SLMSuperPixel



# The global variables for initializing the SLM

# the x and y pixels are the width and height of the SLM
x_pixels = 300
y_pixels = 200

# the physical dimensions of the SLM, both width and height, in astropy units
x_dim = 10 * u.mm
y_dim = 20 * u.mm

# the primary wavelength of the optical system, in astropy units
wavelength = 1.55 * u.micron

# assuming the entrance pupil is circualr or square, the diameter or width of the entrance pupil in SLM pixels
e_diam_pixels = 150

# the focal length of the lens to the image plane, in astropy units
focal_length = 10 * u.mm

# the max phase that the SLM is able to apply, assuming the minimum is 0 shift.
max_phase_shift = 2 * np.pi

# wether or not to only apply patterns to the entrance pupil area
only_in_e_diam = True

# the number of pixels to combine together to form the super pixels, 
# each forms a 2 by 2 checkerboard pattern so has to be a multiple of 2
pix_per_super = 8

# wether or not the max phase shift is less than 2 pi
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

# sets the amplitude to a given value between 0 and 1
flat_ampl_value = 0.5
test_slm.flat_ampl(flat_ampl_value)

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