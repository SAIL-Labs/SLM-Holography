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

These functions apply a flat phase and amplitude to the SLM.
This also shows and explains all of the variables when creating the class

#### [Example 2](2-gaussian_ampl.py)

This function applies a gaussian to the amplitude only.

#### [Example 3](3-focal_plane_patterns.py)

This function creates a circular array and then requests it in the focal plane.  This array is requested with a certain pixel scale and total intensity.

## API Reference

### SLMSuperPixel class

```
SLMSuperPixel(x_pixels, y_pixels, x_dim, y_dim, wavelength, e_diam_pixels, focal_length = 1 * u.mm, max_phase_shift = 4*np.pi, only_in_e_diam = True, pix_per_super = 2, less_than_2pi = False)
```

**parameters:**

x\_pixels : int

- the width of the slm in pixels

y\_pixels : int

 - the height of the slm in pixels

x\_dim : astropy.units convertable to meters

 - the width of the slm in meters

y\_dim : astropy.units convertable to meters

 - the height of the slm in meters

wavelength : astropy.units convertable to meters
	
 - the wavelength of light that the SLM is calibrated for

e\_diam\_pixels : int, optional
	
  - the diameter or width of the entrance pupil in pixels (assuming a circular or square entrance pupil)

focal_length : astropy.units convertable to meters, optional

 - if the SLM is being measured in the focal plane, the effective focal length of the lens

max_phase_shift : float, optional

 - the maximum phase shift that the SLM can apply, assuming that it ranges from \[0,max_phase_shift\]

only_in_e_diam : bool, optional
	
  - wether or not to only apply patterns to the area covered by the entrance pupil, shrinks the arrays to have dimensions (e\_diam\_pixels//pix_per_super + 1, e\_diam\_pixels//pix_per_super + 1)

pix_per_super : int, even, optional

 - how many pixels to group together to form a super pixel, must be an even integer greater than or equal to 2

less_than_2pi : bool, optional

 - wether or not the SLM has a total phase range that is smaller than 2pi

 - also determines if phase wrapping should be used, if False phase wrapping will occur, if True phase wrapping wil not occur 

### SLMSuperPixel.double_pixel_convert

```
SLMSuperPixel.double_pixel_convert(self, add_padding = True, **kwargs)
```

This function takes whatever is in the amplitude and phase arrays and encodes it into the SLM pixel scale.  The returned array has dimensions of (y_pixels, x_pixels) or (e\_diam\_pixels, e\_diam\_pixels)

**parameters:**

add_padding : bool, optional

- if True, the returned array is padded to make its dimensions (y_pixels, x_pixels)
- if False, the returned array is not padded and has dimensions (e\_diam\_pixels, e\_diam\_pixels)

\*\*kwargs : the arguments passed to the padding function, optional

**returns:**

an array with sepcified dimensions, with each value having a phase in radians ranging from \[0,max_phase_shift\]

