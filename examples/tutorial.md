# How to use SLM_super_pixel_holography

first, import the SLMSuperPixel class

```
from SLM_super_pixel_holography import SLMSuperPixel
```

This class has 3 steps, creating the object, adjusting the amplitude and phase, and finally encoding it into the SLM pixel scale.


### SLMSuperPixel class

```
SLMSuperPixel(x_pixels, y_pixels, x_dim, y_dim, wavelength, e_diam_pixels, focal_length = 1 * u.mm, max_phase_shift = 4*np.pi, only_in_e_diam = True, pix_per_super = 2, less_than_2pi = False)
```

	x\_pixels : int
		the width of the slm in pixels
	y\_pixels : int
		the height of the slm in pixels
	x\_dim : astropy.units convertable to meters
		the width of the slm in meters
	y\_dim : astropy.units convertable to meters
		the height of the slm in meters
	wavelength : astropy.units convertable to meters
		the wavelength of light that the SLM is calibrated for
	e\_diam\_pixels : int, optional
		the diameter or width of the entrance pupil in pixels (assuming a circular or square entrance pupil)
	focal_length : astropy.units convertable to meters, optional
		if the SLM is being measured in the focal plane, the effective focal length of the lens
	max_phase_shift : float, optional
		the maximum phase shift that the SLM can apply, assuming that it ranges from [0,max_phase_shift]
	only_in_e_diam : bool, optional
		wether or not to only apply patterns to the area covered by the entrance pupil, shrinks the arrays to have dimensions 
		(e\_diam\_pixels//pix_per_super + 1, e\_diam\_pixels//pix_per_super + 1)
	pix_per_super : int, even, optional
		how many pixels to group together to form a super pixel, must be an even integer greater than or equal to 2
	less_than_2pi : bool, optional
		wether or not the SLM has a total phase range that is smaller than 2pi
		Also determines if phase wrapping should be used, if False phase wrapping will occur, if True phase wrapping wil not occur 

