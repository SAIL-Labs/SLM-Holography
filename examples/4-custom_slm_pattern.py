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


# creates a pattern to apply to the image plane or focal plane
# this example is a filled in circle with phase ramp as the radius increases

dimensions = test_slm.get_array_dimensions()

Y, X = np.mgrid[-dimensions[0]//2:(dimensions[0]//2), -dimensions[1]//2:(dimensions[1]//2)]
R = np.sqrt(Y**2 + X**2)

focal_array = np.zeros(dimensions, dtype = 'complex')
focal_array[R < 50] = 0.5 * np.exp(1j * R[R < 50])


test_slm.custom_complex(focal_array)

# converts the pattern to the SLM using the double pixel method and returns an array that is in radians
# if the pattern is only encoded in the entrance pupil, if add_padding is true the array is padded to be the size of the SLM
add_padding = False
# the amplitude and phase for the padding
padding_phase = 0
padding_ampl = 1

slm_final_array = test_slm.double_pixel_convert(add_padding, 
                                                ampl_padding = padding_ampl, 
                                                phase_padding = padding_phase)


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
