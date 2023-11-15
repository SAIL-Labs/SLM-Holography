# How to use SLM_super_pixel_holography

first, import the SLMSuperPixel class

```
from SLM_super_pixel_holography import SLMSuperPixel
```

This class has 3 steps, creating the object, adjusting the amplitude and phase, and finally encoding it into the SLM pixel scale.

The variables for creating the object is described in the [SLMSuperPixel class](https://github.com/SAIL-Labs/SLM-Holography/edit/main/examples/tutorial.md#slmsuperpixel-class) section.

With an object created, the values of the amplitude and phase arrays can be set using numerous functions.

## Examples

#### [Example 1](1-flat_ampl_and_phase.py)

This example applies a flat phase and amplitude to the SLM.
This also shows and explains all of the variables when creating the class.

#### [Example 2](2-flat_resize_and_reshape.py)

This example applies a Gaussian to the amplitude only.

#### [Example 3](3-gaussian_ampl.py)

This example creates a Gaussian pattern and applies it to the center of the SLM

#### [Example 4](4-custom_slm_pattern.py)

This example creates a custom array (circular amplitude with a phase ramp from the center).

#### [Example 5](5-zernike_terms.py)

This example applies Zernike terms to the phase of the SLM array.

#### [Example 6](6-focal_spots.py)

This example creates multiple spots in the focal plane.

#### [Example 7](7-checkerboard_phase.py)

This example creates a checkerboard pattern in the phase of the SLM.

#### [Example 8](8-focal_plane_patterns.py)

This example creates a circular array and then requests it in the focal plane.  This array is requested with a certain pixel scale and total intensity.

#### [Example 9](9-LP_modes.py)

This example creates LP modes for a specific top hat multimode fiber.


## API Reference

### SLMSuperPixel class

```
SLMSuperPixel(x_pixels, y_pixels, x_dim, y_dim, wavelength, e_diam_pixels, focal_length = 1 * u.mm, max_phase_shift = 4*np.pi, only_in_e_diam = True, pix_per_super = 2, less_than_2pi = False)
```

**parameters:**

x\_pixels : int

- The width of the slm in pixels.

y\_pixels : int

 - The height of the slm in pixels.

x\_dim : astropy.units convertable to meters

 - The width of the slm in meters.

y\_dim : astropy.units convertable to meters

 - The height of the slm in meters.

wavelength : astropy.units convertable to meters

 - The wavelength of light that the SLM is calibrated for.

e\_diam\_pixels : int, optional
	
  - The diameter or width of the entrance pupil in pixels (assuming a circular or square entrance pupil).

focal_length : astropy.units convertable to meters, optional

 - If the SLM is being measured in the focal plane, the effective focal length of the lens.

max_phase_shift : float, optional

 - The maximum phase shift that the SLM can apply, assuming that it ranges from \[0,max_phase_shift\].

only_in_e_diam : bool, optional
	
  - Whether or not to only apply patterns to the area covered by the entrance pupil, shrinks the arrays to have dimensions (e\_diam\_pixels//pix_per_super + 1, e\_diam\_pixels//pix_per_super + 1).

pix_per_super : int, even, optional

 - How many pixels to group together to form a super pixel, must be an even integer greater than or equal to 2.

less_than_2pi : bool, optional

 - Whether or not the SLM has a total phase range that is smaller than 2pi.

 - Also determines if phase wrapping should be used, if False phase wrapping will occur, if True phase wrapping wil not occur.

### SLMSuperPixel.double_pixel_convert

```
SLMSuperPixel.double_pixel_convert(self, add_padding = True, **kwargs)
```

This function takes whatever is in the amplitude and phase arrays and encodes it into the SLM pixel scale.  The returned array has dimensions of (y_pixels, x_pixels) or (e\_diam\_pixels, e\_diam\_pixels).

**parameters:**

add_padding : bool, optional

- If True, the returned array is padded to make its dimensions (y_pixels, x_pixels).
- If False, the returned array is not padded and has dimensions (e\_diam\_pixels, e\_diam\_pixels).

\*\*kwargs : the arguments passed to the padding function, optional

**returns:**

An array with sepcified dimensions, with each value having a phase in radians ranging from \[0,max_phase_shift\].

### SLMSuperPixel.get_array_dimensions

```
SLMSuperPixel.get_array_dimensions()
```

**returns:**

pixels_y, pixels_x

- The height and width of the amplitude and phase arrays.

### SLMSuperPixel.add_padding

```
SLMSuperPixel.add_padding(ampl_padding = 0, phase_padding = 0, verbose = False)
```

**parameters**

ampl_padding : float, optional

 - the amplitude values to pad to the array, must be between \[0,1\].

phase_padding : float, optional

 - the phase values to pad to the array, must be between \[0, max_phase_shift\].

verbose : bool, optional

 - whether or not to be verbose.

### SLMSuperPixel.image_shift

```
SLMSuperPixel.image_shift(x_shift, y_shift, shift_super_pixel_array = True)
```

x_shift : int

- The number of pixels to shift in the x direction.

y_shift : int

 - The number of pixels to shift in the y direction.

shift_super_pixel_array : bool, optional

- If True, the super pixel arrays get shifted by x_shift, y_shift.
- If False, the final slm pixel array gets shifted by x_shift, y_shift.

### SLMSuperPixel.custom_ampl
```
SLMSuperPixel.custom_ampl(custom_ampl)
```

**parameters:**

custom_ampl : {(pixels_y, pixels_x)}array_like

- An array to set the SLM amplitude array to, should have dimensions equal to the dimensions of the SLM amplitude array.

### SLMSuperPixel.custom_phase
```
SLMSuperPixel.custom_phase(custom_phase)
```

**parameters:**

custom_phase : {(pixels_y, pixels_x)}array_like

- An array to set the SLM phase array to, should have dimensions equal to the dimensions of the SLM phase array.

### SLMSuperPixel.custom_complex
```
SLMSuperPixel.custom_complex(custom_complex)
```

**parameters:**

custom_phase : {(pixels_y, pixels_x)}array_like, complex

- A complex array to set the amplitude and phase arrays to.  The amplitude array is the np.abs(custom complex), and the phase array is the np.angle(custom complex).
- Should have dimensions equal to the dimensions of the SLM phase array.
