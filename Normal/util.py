import numpy as np

from quadtree import QuadTree


def in_main_body(x, y):
    """
    Checks if the given point (x, y) is inside the main cardioid of the Mandelbrot set.

    Parameters:
    - x (float): The x-coordinate of the point.
    - y (float): The y-coordinate of the point.

    Returns:
    bool: True if the point is inside the main cardioid, False otherwise.
    """

    q = (x - 0.25) * (x - 0.25) + y * y
    return q * (q + x - 0.25) <= 0.25 * y * y


def calculate(x0,
              y0,
              max_iterations,
              escape_radius,
              smooth,
              color_scheme,
              num_computed,
              period_checking,
              memo=None):
    """
    Performs Mandelbrot set iteration to calculate the color of a given point.

    Parameters:
    - x0 (float): The x-coordinate of the point.
    - y0 (float): The y-coordinate of the point.
    - max_iterations (int): The maximum number of iterations.
    - escape_radius (float): The escape radius for determining if a point is in the Mandelbrot set.
    - smooth (bool): Whether to use smooth coloring.
    - color_scheme (function): A function that maps Mandelbrot set parameters to a color.
    - num_computed (int): The number of points already computed.
    - period_checking (bool): Whether to perform periodicity checking.
    - memo (dict): A memoization dictionary for caching computed points.

    Returns:
    Tuple[np.ndarray, bool]: A tuple containing the calculated color and a boolean indicating if the point is in the set.
    """

    if memo is not None and (x0, y0) in memo:
        return memo[(x0, y0)]

    num_computed += 1

    if in_main_body(x0, y0):
        color = color_scheme(max_iteration=max_iterations, iteration=max_iterations,
                             distance_estimate=None, final=(x0, y0), escape_radius=escape_radius,
                             smooth=smooth)

        if memo is not None:
            memo[(x0, y0)] = (color, True)

        return color, True

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

    while x2 + y2 <= escape_radius_squared and iterations < max_iterations:
        x = x2 - y2 + x0
        y = w - x2 - y2 + y0
        x2 = x * x
        y2 = y * y
        w = (x + y) * (x + y)

        dx2 = 2 * (x * dx - y * dy) + 1
        dy = 2 * (y * dx + x * dy)
        dx = dx2

        iterations += 1

        if period_checking:
            if x_old == x and y_old == y:
                iterations = max_iterations
                break
            period += 1

            if period > 20:
                period = 0
                x_old = x
                y_old = y

    z = x2 + y2
    dz = dx * dx + dy * dy

    distance_estimate = np.log(
        z) * np.sqrt(z / dz) if iterations != max_iterations else None

    color = color_scheme(max_iteration=max_iterations,
                         iteration=iterations,
                         distance_estimate=distance_estimate,
                         final=(x, y),
                         escape_radius=escape_radius, smooth=smooth)

    if memo is not None:
        memo[(x0, y0)] = (color, iterations == max_iterations)
    return color, iterations == max_iterations


def calculate_quadtree(quad_tree: QuadTree,
                       pixels: np.ndarray,
                       x: np.ndarray,
                       y: np.ndarray,
                       max_iterations,
                       escape_radius,
                       smooth,
                       color_scheme,
                       num_computed,
                       period_checking,
                       memo=None):
    """
    Calculates the Mandelbrot set for a given QuadTree region.

    Parameters:
    - quad_tree (QuadTree): The QuadTree region to calculate.
    - pixels (np.ndarray): The pixel array to store the calculated colors.
    - x (np.ndarray): The x-coordinates of the points.
    - y (np.ndarray): The y-coordinates of the points.
    - max_iterations (int): The maximum number of iterations.
    - escape_radius (float): The escape radius for determining if a point is in the Mandelbrot set.
    - smooth (bool): Whether to use smooth coloring.
    - color_scheme (function): A function that maps Mandelbrot set parameters to a color.
    - num_computed (int): The number of points already computed.
    - period_checking (bool): Whether to perform periodicity checking.
    - memo (dict): A memoization dictionary for caching computed points.

    Returns:
    Tuple[bool, np.ndarray]: A tuple containing a boolean indicating if the QuadTree region is splittable
                             and the border color if not splittable.
    """

    tl = quad_tree.tl
    br = quad_tree.br

    if quad_tree.rows == 1 and quad_tree.cols == 1:
        color, _ = calculate(x[tl[0]], y[tl[1]], max_iterations, escape_radius, smooth, color_scheme, num_computed,
                             period_checking, memo)
        return False, color

    split = False
    border = None

    for i in range(tl[0], br[0]):
        if border is not None:
            top_color, _ = calculate(x[i], y[tl[1]], max_iterations, escape_radius, smooth, color_scheme, num_computed,
                                     period_checking, memo)

            if not split:
                split |= (border != top_color).any()

            pixels[tl[1]][i] = top_color

            bottom_color, _ = calculate(x[i], y[br[1] - 1], max_iterations, escape_radius, smooth, color_scheme,
                                        num_computed, period_checking, memo)
            if not split:
                split |= (border != bottom_color).any()

            pixels[br[1] - 1][i] = bottom_color
        else:
            top_color, _ = calculate(x[i], y[tl[1]], max_iterations, escape_radius, smooth, color_scheme, num_computed,
                                     period_checking, memo)
            pixels[tl[1]][i] = top_color
            border = top_color

            bottom_color, _ = calculate(x[i], y[br[1] - 1], max_iterations, escape_radius, smooth, color_scheme,
                                        num_computed, period_checking, memo)
            split |= (border != bottom_color).any()
            pixels[br[1] - 1][i] = bottom_color

    for j in range(tl[1], br[1]):
        left_color, _ = calculate(x[tl[0]], y[j], max_iterations, escape_radius, smooth, color_scheme, num_computed,
                                  period_checking, memo)
        pixels[j][tl[0]] = left_color

        if not split:
            split |= (left_color != border).any()

        right_color, _ = calculate(x[br[0] - 1], y[j], max_iterations, escape_radius, smooth, color_scheme,
                                   num_computed, period_checking, memo)
        pixels[j][br[0] - 1] = right_color

        if not split:
            split |= (right_color != border).any()

    return split and quad_tree.cols > 3 and quad_tree.rows > 3, border


def calculated_mixed_raster_quadtree(quad_tree: QuadTree, pixels: np.ndarray, x: np.ndarray, y: np.ndarray,
                                     max_iterations,
                                     escape_radius,
                                     smooth,
                                     color_scheme,
                                     num_computed,
                                     period_checking, memo=None):
    """
    Calculates the Mandelbrot set using a mixed raster and quadtree approach.

    Parameters:
    - quad_tree (QuadTree): The QuadTree region to calculate.
    - pixels (np.ndarray): The pixel array to store the calculated colors.
    - x (np.ndarray): The x-coordinates of the points.
    - y (np.ndarray): The y-coordinates of the points.
    - max_iterations (int): The maximum number of iterations.
    - escape_radius (float): The escape radius for determining if a point is in the Mandelbrot set.
    - smooth (bool): Whether to use smooth coloring.
    - color_scheme (function): A function that maps Mandelbrot set parameters to a color.
    - num_computed (int): The number of points already computed.
    - period_checking (bool): Whether to perform periodicity checking.
    - memo (dict): A memoization dictionary for caching computed points.

    Returns:
    Tuple[bool, np.ndarray]: A tuple containing a boolean indicating if the QuadTree region is splittable
                             and the border color if not splittable.
    """

    tl = quad_tree.tl
    br = quad_tree.br

    if quad_tree.rows == 1 and quad_tree.cols == 1:
        color, _ = calculate(x[tl[0]], y[tl[1]], max_iterations,
                             escape_radius,
                             smooth,
                             color_scheme,
                             num_computed,
                             period_checking, memo)
        return False, color

    color = np.array([-1.0, -1.0, -1.0])

    isMandelbrot = True
    hasMandelbrot = False

    for i in range(tl[0], br[0]):
        border, inSet = calculate(x[i], y[tl[1]], max_iterations,
                                  escape_radius,
                                  smooth,
                                  color_scheme,
                                  num_computed,
                                  period_checking, memo)

        if not hasMandelbrot and inSet:
            color = border

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        pixels[tl[1]][i] = border

        border, inSet = calculate(x[i], y[br[1] - 1], max_iterations,
                                  escape_radius,
                                  smooth,
                                  color_scheme,
                                  num_computed,
                                  period_checking, memo)

        if not hasMandelbrot and inSet:
            color = border

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        pixels[br[1] - 1][i] = border

    for j in range(tl[1], br[1]):

        border, inSet = calculate(x[tl[0]], y[j], max_iterations,
                                  escape_radius,
                                  smooth,
                                  color_scheme,
                                  num_computed,
                                  period_checking, memo)

        if not hasMandelbrot and inSet:
            color = border

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        pixels[j][tl[0]] = border

        border, inSet = calculate(x[br[0] - 1], y[j], max_iterations,
                                  escape_radius,
                                  smooth,
                                  color_scheme,
                                  num_computed,
                                  period_checking, memo)

        if not hasMandelbrot and inSet:
            color = border

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        pixels[br[0] - 1][j] = border

    return isMandelbrot != hasMandelbrot and (quad_tree.rows > 3 and quad_tree.cols > 3), color


def row_raster(pixels: np.ndarray,
               row: int,
               x: np.ndarray,
               y: np.ndarray,
               max_iterations,
               escape_radius,
               smooth,
               color_scheme,
               num_computed,
               period_checking, memo=None):
    """
    Calculates the Mandelbrot set for a specific row using a raster approach.

    Parameters:
    - pixels (np.ndarray): The pixel array to store the calculated colors.
    - row (int): The row index to calculate.
    - x (np.ndarray): The x-coordinates of the points.
    - y (np.ndarray): The y-coordinates of the points.
    - max_iterations (int): The maximum number of iterations.
    - escape_radius (float): The escape radius for determining if a point is in the Mandelbrot set.
    - smooth (bool): Whether to use smooth coloring.
    - color_scheme (function): A function that maps Mandelbrot set parameters to a color.
    - num_computed (int): The number of points already computed.
    - period_checking (bool): Whether to perform periodicity checking.
    - memo (dict): A memoization dictionary for caching computed points.

    Returns:
    int: The number of filled pixels in the row.
    """

    unfilled_pixel = np.array([-1.0, -1.0, -1.0])
    filled = 0

    for j, x_val in enumerate(x):
        if (pixels[row][j] != unfilled_pixel).any():
            color, _ = calculate(x_val, y[row], max_iterations,
                                 escape_radius,
                                 smooth,
                                 color_scheme,
                                 num_computed,
                                 period_checking, memo)
            pixels[row][j] = color
            filled += 1

    return filled
