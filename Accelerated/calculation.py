import numpy as np
from numba import njit, f8, i4, u8, b1, u1


from .coloring import grayscale, simple_hsv, quilt_coloring, colormap_coloring


@njit(fastmath=True)
def in_main_body(x, y):
    """
    Checks if the given coordinates (x, y) are inside the main cardioid or period-2 bulb of the Mandelbrot set.

    Parameters:
    - x (float): The x-coordinate in the complex plane.
    - y (float): The y-coordinate in the complex plane.

    Returns:
    - bool: True if inside the main cardioid or period-2 bulb, False otherwise.
    """

    q = (x - 0.25) * (x - 0.25) + y * y
    return q * (q + x - 0.25) <= 0.25 * y * y


@njit(u1[:](u8, i4, f8, f8, f8, f8, b1, u1, u1[:, :]), fastmath=True)
def determine_colorscheme(max_iteration: int,
                          iteration: int,
                          distance_estimate: float,
                          x,
                          y,
                          escape_radius: float,
                          smooth: bool,
                          color_scheme: int,
                          color_map):
    """
    Determines the color for a pixel based on the specified color scheme.

    Parameters:
    - max_iteration (int): The maximum number of iterations.
    - iteration (int): The current iteration count.
    - distance_estimate (float): The distance estimate from the Mandelbrot set.
    - x (float): The x-coordinate in the complex plane.
    - y (float): The y-coordinate in the complex plane.
    - escape_radius (float): The escape radius for the Mandelbrot set.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Identifier for the color scheme.
    - color_map (numpy.ndarray): Color map for mapping fractal values to colors.

    Returns:
    - numpy.ndarray: An array representing the color for the pixel.
    """

    if color_scheme == 0:
        return grayscale(max_iteration,
                          iteration,
                          x,
                          y,
                          escape_radius,
                          smooth)
    elif color_scheme == 1:
        return simple_hsv(max_iteration,
                           iteration,
                           x,
                           y,
                           escape_radius,
                           smooth)
    elif color_scheme == 2:
        return quilt_coloring(max_iteration,
                               iteration,
                               x, y,
                               escape_radius,
                               distance_estimate,
                               smooth, )
    elif color_scheme == 3:
        return colormap_coloring(max_iteration,
                                  iteration,
                                  x,
                                  y,
                                  escape_radius,
                                  smooth,
                                  color_map)
    else:
        return grayscale(max_iteration,
                          iteration,
                          x, y,
                          escape_radius,
                          smooth)


@njit(u1[:](f8, f8, u8, f8, b1, u1, u1[:, :], b1), fastmath=True)
def calculate(x0: float,
              y0: float,
              max_iterations: int,
              escape_radius: float,
              smooth: bool,
              color_scheme: int,
              color_map,
              period_checking: bool):
    """
    Calculates the Mandelbrot fractal value for a given point in the complex plane.

    Parameters:
    - x0 (float): The x-coordinate of the point in the complex plane.
    - y0 (float): The y-coordinate of the point in the complex plane.
    - max_iterations (int): The maximum number of iterations.
    - escape_radius (float): The escape radius for the Mandelbrot set.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Identifier for the color scheme.
    - color_map (numpy.ndarray): Color map for mapping fractal values to colors.
    - period_checking (bool): Flag indicating whether to perform period checking.

    Returns:
    - numpy.ndarray: An array representing the color value for the pixel.
    """

    if in_main_body(x0, y0):
        color = determine_colorscheme(max_iteration=max_iterations, iteration=max_iterations,
                                     distance_estimate=0.0, x=x0, y=y0, escape_radius=escape_radius,
                                     smooth=smooth, color_scheme=color_scheme, color_map=color_map)

        return np.array([1, color[0], color[1], color[2]], dtype=np.uint8)

    x = 0
    y = 0
    x2 = 0
    y2 = 0
    w = 0

    x_old = 0
    y_old = 0
    period = 0

    dx = 0
    dy = 0

    iterations = 0

    escape_radius_squared = escape_radius * escape_radius

    for i in range(max_iterations):
        x = x2 - y2 + x0
        y = w - x2 - y2 + y0
        x2 = x * x
        y2 = y * y
        w = (x + y) * (x + y)

        dx2 = 2 * (x * dx - y * dy) + 1
        dy = 2 * (y * dx + x * dy)
        dx = dx2

        iterations += 1

        if x2 + y2 >= escape_radius_squared:
            break

        if period_checking:
            if x_old == x and y_old == y:
                iterations = max_iterations
                break
            period += 1

            if period > period_checking:
                period = 0
                x_old = x
                y_old = y

    z = x2 + y2
    dz = dx * dx + dy * dy

    distance_estimate = 0.0

    if iterations != max_iterations:
        distance_estimate = np.log(z) * np.sqrt(z / dz)

    color = determine_colorscheme(max_iteration=max_iterations,
                                  iteration=iterations,
                                  distance_estimate=distance_estimate,
                                  x=x, y=y,
                                  escape_radius=escape_radius,
                                  smooth=smooth,
                                  color_scheme=color_scheme,
                                  color_map=color_map)

    isMaxIteration = 0

    if iterations == max_iterations:
        isMaxIteration = 1

    return np.array([isMaxIteration, color[0], color[1], color[2]], dtype=np.uint8)

