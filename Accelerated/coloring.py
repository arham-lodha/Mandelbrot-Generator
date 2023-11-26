import numpy as np
from numba import njit, float64, int32, b1, u1


@njit(float64(float64, float64, int32, float64), fastmath=True)
def continous_dwell(x: float, y: float, dwell: int, escape_radus: float) -> float:
    """
        Calculate continuous dwell value for smooth coloring.

        Parameters:
        - x (float): Real component of the complex number.
        - y (float): Imaginary component of the complex number.
        - dwell (int): Iteration count.
        - escape_radus (float): Escape radius for the fractal.

        Returns:
        - float: Continuous dwell value.
        """
    return dwell - np.log2(np.log2(x * x + y * y)) + np.log2(np.log2(escape_radus)) + 1


@njit(u1[:](int32, int32, float64, float64, float64, b1, u1[:, :]), fastmath=True)
def colormap_coloring(max_iteration,
                      iteration,
                      final_x,
                      final_y,
                      escape_radius,
                      smooth,
                      color_map):
    """
        Apply colormap coloring to determine the RGB color for a pixel.

        Parameters:
        - max_iteration (int): Maximum number of iterations.
        - iteration (int): Current iteration count.
        - final_x (float): Final x-coordinate in the complex plane.
        - final_y (float): Final y-coordinate in the complex plane.
        - escape_radius (float): Escape radius for the fractal.
        - smooth (bool): Flag indicating whether to use smooth coloring.
        - color_map (numpy.ndarray): Color map for mapping fractal values to colors.

        Returns:
        - numpy.ndarray: RGB color for the pixel.
    """

    i = iteration
    if iteration != max_iteration and smooth != 0:
        i = continous_dwell(final_x, final_y, iteration, escape_radius)

    N = color_map.shape[0]

    i = int(min(np.mod(i, N), np.mod(i, max_iteration)))

    return color_map[i]  # Ensure consistent data type


@njit(u1[:](int32, int32, float64, float64, float64, b1), fastmath=True)
def simple_hsv(max_iteration,
               iteration,
               final_x,
               final_y,
               escape_radius,
               smooth):
    """
    Apply simple HSV coloring to determine the HSV color for a pixel.

    Parameters:
    - max_iteration (int): Maximum number of iterations.
    - iteration (int): Current iteration count.
    - final_x (float): Final x-coordinate in the complex plane.
    - final_y (float): Final y-coordinate in the complex plane.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.

    Returns:
    - numpy.ndarray: RGB color for the pixel.
    """

    i = iteration
    if smooth != 0 and iteration != max_iteration:
        i = continous_dwell(final_x, final_y, iteration, escape_radius)

    hue = 255
    value = 0
    if iteration != max_iteration:
        hue = np.round(np.power(i / max_iteration * 255, 2)) % 255
        value = 255

    return np.array([hue, 255, value], dtype=np.uint8)  # Ensure consistent data type


@njit(u1[:](int32, int32, float64, float64, float64, b1), fastmath=True)
def grayscale(max_iteration,
              iteration: int,
              final_x,
              final_y,
              escape_radius: float,
              smooth: bool):
    """
    Apply grayscale coloring to determine the RGB color for a pixel.

    Parameters:
    - max_iteration (int): Maximum number of iterations.
    - iteration (int): Current iteration count.
    - final_x (float): Final x-coordinate in the complex plane.
    - final_y (float): Final y-coordinate in the complex plane.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.

    Returns:
    - numpy.ndarray: RGB color for the pixel.
    """

    if iteration != max_iteration and smooth:
        iteration = continous_dwell(final_x, final_y, iteration, escape_radius)

    s = np.sqrt(iteration / max_iteration)

    return np.array([s * 255, s * 255, s * 255], dtype=np.uint8)  # Ensure consistent data type


@njit(u1[:](int32, int32, float64, float64, float64, float64, b1), fastmath=True)
def quilt_coloring(max_iteration,
                   iteration,
                   final_x,
                   final_y,
                   escape_radius,
                   distance_estimate,
                   smooth):
    """
    Apply quilt coloring to determine the HSV color for a pixel.

    Parameters:
    - max_iteration (int): Maximum number of iterations.
    - iteration (int): Current iteration count.
    - final_x (float): Final x-coordinate in the complex plane.
    - final_y (float): Final y-coordinate in the complex plane.
    - escape_radius (float): Escape radius for the fractal.
    - distance_estimate (float): Distance estimate from the Mandelbrot set.
    - smooth (bool): Flag indicating whether to use smooth coloring.

    Returns:
    - numpy.ndarray: RGB color for the pixel.
    """

    color = np.array([255, 255, 0], dtype=np.uint8)

    if max_iteration != iteration:
        x = final_x
        y = final_y

        fin_angle = np.arctan2(y, x)

        fin_radius = 0.0
        if smooth != 0:
            fin_radius = continous_dwell(x, y, iteration, escape_radius) - iteration

        dscale = np.log(distance_estimate / 0.00001)

        if dscale > 0:
            value = 1
        elif dscale > -8:
            value = (8 + dscale) / 8
        else:
            value = 0

        p = np.log(iteration) / np.log(100000)

        if p < 0.5:
            p = 1.0 - 1.5 * p
            angle = 1 - p
            radius = np.sqrt(p)
        else:
            p = 1.5 * p - 0.5
            angle = p
            radius = np.sqrt(p)

        if iteration % 2 == 0:
            value = 0.85 * value
            radius = 0.667 * radius

        if fin_angle < 0:
            angle += 0.02

        angle += 0.0001 * fin_radius

        hue = angle * 10.0
        hue = hue - np.floor(hue)

        saturation = radius - np.floor(radius)

        color = np.round(np.array([hue * 255, saturation * 255, value * 255], dtype=np.float64)).astype(
            np.uint8)
    return color
