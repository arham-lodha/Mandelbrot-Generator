from numba import njit, u1, i4, f8, u8, b1, prange
import numpy as np

from .calculation import calculate


@njit(u1[:](i4[:], i4[:], u1[:, :, :], b1[:, :], f8[:], f8[:], u8, f8, b1, u1, u1[:, :], b1))
def calculate_mixed(tl, br,
                    pixels,
                    seen,
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
    Calculate mixed quadtree within the specified region.

    Parameters:
    - tl (numpy.ndarray): Top-left coordinates of the region.
    - br (numpy.ndarray): Bottom-right coordinates of the region.
    - pixels (numpy.ndarray): Array to store the calculated fractal values.
    - seen (numpy.ndarray): Array to track pixels that have been calculated.
    - x (numpy.ndarray): Array of x-coordinates in the complex plane.
    - y (numpy.ndarray): Array of y-coordinates in the complex plane.
    - max_iterations (int): Maximum number of iterations.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Fractal coloring scheme.
    - color_map: Color map for mapping fractal values to colors.
    - period_checking (bool): Flag indicating whether to perform period checking.

    Returns:
    - numpy.ndarray: Result array containing information about split, Mandelbrot membership, and border color.
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

        seen[tl[1]][tl[0]] = True

        return np.array([0, color[0], color[1], color[2]], dtype=np.uint8)

    isMandelbrot = True
    hasMandelbrot = False
    border = np.array([0, 0, 0], dtype=np.uint8)

    for i in range(tl[0], br[0]):
        top = calculate(x[i],
                        y[tl[1]],
                        max_iterations,
                        escape_radius,
                        smooth,
                        color_scheme,
                        color_map,
                        period_checking)

        inSet = top[0] == 1
        top_color = top[1:]

        if not hasMandelbrot and inSet:
            border = top_color

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        pixels[tl[1]][i] = top_color

        bottom = calculate(x[i],
                           y[br[1] - 1],
                           max_iterations,
                           escape_radius,
                           smooth,
                           color_scheme,
                           color_map,
                           period_checking)

        inSet = bottom[0] == 1
        bottom_color = bottom[1:]

        if not hasMandelbrot and inSet:
            border = bottom_color

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        pixels[br[1] - 1][i] = bottom_color

        seen[tl[1]][i] = True
        seen[br[1] - 1][i] = True

    for j in range(tl[1], br[1]):
        left = calculate(x[tl[0]],
                         y[j],
                         max_iterations,
                         escape_radius,
                         smooth,
                         color_scheme,
                         color_map,
                         period_checking)

        inSet = left[0] == 1
        left_color = left[1:]

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        if not hasMandelbrot and inSet:
            border = left_color

        pixels[j][tl[0]] = left_color

        right = calculate(x[br[0] - 1],
                          y[j],
                          max_iterations,
                          escape_radius,
                          smooth,
                          color_scheme,
                          color_map, period_checking)

        inSet = right[0] == 1
        right_color = right[1:]

        isMandelbrot &= inSet
        hasMandelbrot |= inSet

        if not hasMandelbrot and inSet:
            border = right_color

        pixels[j][br[0] - 1] = right_color

        seen[j][tl[0]] = True
        seen[j][br[0] - 1] = True

    split_val = 0

    if isMandelbrot != hasMandelbrot and cols >= 3 and rows >= 3:
        split_val = 1

    mandelbrot_val = 1
    if not isMandelbrot:
        mandelbrot_val = 0

    return np.array([split_val, mandelbrot_val, border[0], border[1], border[2]], dtype=np.uint8)


@njit(u1[:, :](i4[:, :, :], u1[:, :, :], b1[:, :], f8[:], f8[:], u8, f8, b1, u1, u1[:, :], b1), parallel=True)
def fast_mixed_quadtree(intervals, pixels, seen, x,
                        y,
                        max_iterations: int,
                        escape_radius: float,
                        smooth: bool,
                        color_scheme: int,
                        color_map,
                        period_checking: bool, ):
    """
    Perform fast parallel computation of mixed fractal values for multiple quadtrees.

    Parameters:
    - intervals (numpy.ndarray): Array of intervals to compute fractal values.
    - pixels (numpy.ndarray): Array to store the calculated fractal values.
    - seen (numpy.ndarray): Array to track pixels that have been calculated.
    - x (numpy.ndarray): Array of x-coordinates in the complex plane.
    - y (numpy.ndarray): Array of y-coordinates in the complex plane.
    - max_iterations (int): Maximum number of iterations.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Fractal coloring scheme.
    - color_map: Color map for mapping fractal values to colors.
    - period_checking (bool): Flag indicating whether to perform period checking.

    Returns:
    - numpy.ndarray: Result array containing information about split, Mandelbrot membership, and border color for each interval.
    """

    results = np.zeros(shape=(intervals.shape[0], 4), dtype=np.uint8)

    for i in prange(intervals.shape[0]):
        results[i] = calculate_mixed(intervals[i][0], intervals[i][1], pixels, seen, x, y, max_iterations, escape_radius,
                                     smooth, color_scheme, color_map, period_checking)

    return results
