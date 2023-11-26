from numba import njit, u1, i4, f8, u8, b1, prange
import numpy as np
from .calculation import calculate


@njit(u1[:](i4[:], i4[:], u1[:, :, :], f8[:], f8[:], u8, f8, b1, u1, u1[:, :], b1))
def calculate_quadtree(tl, br,
                       pixels,
                       x,
                       y,
                       max_iterations: int,
                       escape_radius: float,
                       smooth: bool,
                       color_scheme: int,
                       color_map,
                       period_checking: bool,
                       ):
    """
    Calculate quadtree fractal values within the specified region.

    Parameters:
    - tl (numpy.ndarray): Top-left coordinates of the region.
    - br (numpy.ndarray): Bottom-right coordinates of the region.
    - pixels (numpy.ndarray): Array to store the calculated fractal values.
    - x (numpy.ndarray): Array of x-coordinates in the complex plane.
    - y (numpy.ndarray): Array of y-coordinates in the complex plane.
    - max_iterations (int): Maximum number of iterations.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Fractal coloring scheme.
    - color_map: Color map for mapping fractal values to colors.
    - period_checking (bool): Flag indicating whether to perform period checking.

    Returns:
    - numpy.ndarray: Result array containing information about split, border color.
    """

    cols = br[0] - tl[0]
    rows = br[1] - tl[1]

    if rows == 1 and cols == 1:
        color = calculate(x[tl[0]],
                          y[tl[1]],
                          max_iterations,
                          escape_radius,
                          smooth,
                          color_scheme,
                          color_map,
                          period_checking)[1:]

        return np.array([0, color[0], color[1], color[2]], dtype=np.uint8)

    split = False
    border_set = False
    border = np.array([-1, -1, -1], dtype=np.uint8)

    for i in range(tl[0], br[0]):
        if border_set:
            top_color = calculate(x[i],
                                  y[tl[1]],
                                  max_iterations,
                                  escape_radius,
                                  smooth,
                                  color_scheme,
                                  color_map,
                                  period_checking)[1:]

            if not split and cols >= 3 and rows >= 3:
                split |= not np.array_equal(border, top_color)

            pixels[tl[1]][i] = top_color

            bottom_color = calculate(x[i],
                                     y[br[1] - 1],
                                     max_iterations,
                                     escape_radius,
                                     smooth,
                                     color_scheme,
                                     color_map,
                                     period_checking)[1:]
            if not split and cols >= 3 and rows >= 3:
                split |= not np.array_equal(border, bottom_color)

            pixels[br[1] - 1][i] = bottom_color
        else:
            border = calculate(x[i], y[tl[1]],
                               max_iterations,
                               escape_radius,
                               smooth, color_scheme,
                               color_map,
                               period_checking)[1:]

            pixels[tl[1]][i] = border

            bottom_color = calculate(x[i],
                                     y[br[1] - 1],
                                     max_iterations,
                                     escape_radius, smooth, color_scheme,
                                     color_map,
                                     period_checking)[1:]

            if not split and cols >= 3 and rows >= 3:
                split |= not np.array_equal(border, bottom_color)

            pixels[br[1] - 1][i] = bottom_color

            border_set = True

    for j in range(tl[1], br[1]):
        left_color = calculate(x[tl[0]],
                               y[j],
                               max_iterations,
                               escape_radius,
                               smooth,
                               color_scheme,
                               color_map,
                               period_checking)[1:]
        pixels[j][tl[0]] = left_color

        if not split and cols >= 3 and rows >= 3:
            split |= not np.array_equal(border, left_color)

        right_color = calculate(x[br[0] - 1],
                                y[j],
                                max_iterations,
                                escape_radius,
                                smooth,
                                color_scheme,
                                color_map, period_checking)[1:]
        pixels[j][br[0] - 1] = right_color

        if not split and cols >= 3 and rows >= 3:
            split |= not np.array_equal(border, right_color)

    split_val = 0

    if split:
        split_val = 1

    return np.array([split_val, border[0], border[1], border[2]], dtype=np.uint8)


@njit(u1[:, :](i4[:, :, :], u1[:, :, :], f8[:], f8[:], u8, f8, b1, u1, u1[:, :], b1), parallel=True)
def compute_fast_quadtree(intervals, pixels, x,
                          y,
                          max_iterations: int,
                          escape_radius: float,
                          smooth: bool,
                          color_scheme: int,
                          color_map,
                          period_checking: bool, ):
    """
    Perform fast parallel computation of quadtree fractal values for multiple intervals.

    Parameters:
    - intervals (numpy.ndarray): Array of intervals to compute fractal values.
    - pixels (numpy.ndarray): Array to store the calculated fractal values.
    - x (numpy.ndarray): Array of x-coordinates in the complex plane.
    - y (numpy.ndarray): Array of y-coordinates in the complex plane.
    - max_iterations (int): Maximum number of iterations.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Fractal coloring scheme.
    - color_map: Color map for mapping fractal values to colors.
    - period_checking (bool): Flag indicating whether to perform period checking.

    Returns:
    - numpy.ndarray
    """

    results = np.zeros(shape=(intervals.shape[0], 4), dtype=np.uint8)

    for i in prange(intervals.shape[0]):
        results[i] = calculate_quadtree(intervals[i][0], intervals[i][1], pixels, x, y, max_iterations, escape_radius,
                                        smooth, color_scheme, color_map, period_checking)

    return results
