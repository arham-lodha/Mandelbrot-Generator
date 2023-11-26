import numpy as np

def continous_dwell(x, y, dwell, escape_radus=2):
    """
    Calculate continuous dwell value for a point in the Mandelbrot set.

    Parameters:
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.
    - dwell (int): The dwell value (iteration count).
    - escape_radius (float): The escape radius for determining if a point is in the Mandelbrot set.

    Returns:
    float: Continuous dwell value.
    """

    return dwell - np.log2(np.log2(x * x + y * y)) + np.log2(np.log2(escape_radus)) + 1


def generate_colormap_coloring(colormap, exponetial_cyclic=True):
    """
    Generate a colormap-based coloring function.

    Parameters:
    - colormap (list): A list of colors for the colormap.
    - exponetial_cyclic (bool): Whether to apply exponential cyclic mapping.

    Returns:
    tuple: A tuple containing the colormap coloring function and its mode.
    """

    def colormap_coloring(**kwargs):
        """
        Colormap coloring function for Mandelbrot set.

        Parameters:
        - kwargs (dict): Keyword arguments containing Mandelbrot set parameters.

        Returns:
        numpy.ndarray: RGB color values.
        """

        iteration = kwargs['iteration']
        max_iteration = kwargs['max_iteration']
        escape_radius = kwargs['escape_radius']
        smooth = kwargs['smooth']
        x, y = kwargs['final']

        i = iteration
        if iteration != max_iteration and smooth:
            i = continous_dwell(x, y, iteration, escape_radius)

        N = len(colormap)

        if exponetial_cyclic:
            i = int((np.power(i/max_iteration, 2) * N) % N)
        else:
            i = int(min(i % N, i % max_iteration))

        return np.array(colormap[i])

    return colormap_coloring, "RGB"

def simple_hsv(**kwargs):
    """
    Simple HSV coloring function for Mandelbrot set.

    Parameters:
    - kwargs (dict): Keyword arguments containing Mandelbrot set parameters.

    Returns:
    numpy.ndarray: HSV color values.
    """

    iteration = kwargs['iteration']
    max_iteration = kwargs['max_iteration']
    escape_radius = kwargs['escape_radius']
    smooth = kwargs['smooth']
    x, y = kwargs['final']

    if iteration == max_iteration:
        return np.array([255, 255, 0])

    i = iteration
    if smooth:
        i = continous_dwell(x, y, iteration, escape_radius)

    hue = np.power(i / max_iteration * 360, 1.5) % 255
    value = np.round(255)

    return np.array([hue, 255, value])



def grayscale(**kwargs):
    """
    Grayscale coloring function for Mandelbrot set.

    Parameters:
    - kwargs (dict): Keyword arguments containing Mandelbrot set parameters.

    Returns:
    numpy.ndarray: RGB color values.
    """

    max_iteration = kwargs['max_iteration']
    iteration = kwargs['iteration']
    smooth = kwargs['smooth']
    final = kwargs['final']
    escape_radius = kwargs['escape_radius']

    if iteration != max_iteration and smooth:
        iteration = continous_dwell(final[0], final[1], iteration, escape_radius)

    s = np.sqrt(iteration / max_iteration)

    return np.array([np.round(s * 255), np.round(s * 255), np.round(s * 255)])


def quilt_coloring(**kwargs):
    """
    Quilt coloring function for Mandelbrot set.

    Parameters:
    - kwargs (dict): Keyword arguments containing Mandelbrot set parameters.

    Returns:
    numpy.ndarray: RGB color values.
    """
    iteration = kwargs['iteration']
    distance_estimate = kwargs['distance_estimate']
    smooth = kwargs['smooth']
    x, y = kwargs['final']

    if distance_estimate is None:
        return np.array([255, 255, 0])

    fin_angle = np.arctan2(y, x)

    fin_radius = continous_dwell(x, y, iteration) - iteration if smooth else 0.0

    dscale = np.log2(distance_estimate / 0.00001)

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

    return np.array([np.round(hue * 255), np.round(saturation * 255), np.round(value * 255)])


color_scheme = [
(
        grayscale,
        "RGB"
    ),
(
        simple_hsv, "HSV"
    ),
    (
        quilt_coloring, "HSV"
    )

]
