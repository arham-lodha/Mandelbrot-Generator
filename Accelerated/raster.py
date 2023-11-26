from numba import njit, u1, f8, u8, b1, void, prange
from .calculation import calculate


@njit(void(u1[:, :, :], b1[:, :], f8[:], f8[:], u8, f8, b1, u1, u1[:, :], b1), parallel=True)
def compute_raster(pixels, seen, x, y, max_iterations: int,
                   escape_radius: float,
                   smooth: bool,
                   color_scheme: int,
                   color_map,
                   period_checking: bool):
    """
    Compute raster fractal values for a given set of coordinates.

    Parameters:
    - pixels (numpy.ndarray): Array to store the calculated fractal values.
    - seen (numpy.ndarray): Array indicating whether pixels have been previously computed.
    - x (numpy.ndarray): Array of x-coordinates in the complex plane.
    - y (numpy.ndarray): Array of y-coordinates in the complex plane.
    - max_iterations (int): Maximum number of iterations.
    - escape_radius (float): Escape radius for the fractal.
    - smooth (bool): Flag indicating whether to use smooth coloring.
    - color_scheme (int): Fractal coloring scheme.
    - color_map: Color map for mapping fractal values to colors.
    - period_checking (bool): Flag indicating whether to perform period checking.

    Returns:
    - None

    The function populates the `pixels` array with calculated fractal values based on the given parameters.
    """

    for i in prange(y.shape[0]):
        for j in range(x.shape[0]):
            if seen.shape[1] != 0 and not seen[i, j]:
                pixels[i, j] = calculate(x[j], y[i], max_iterations, escape_radius, smooth, color_scheme, color_map,
                                         period_checking)[1:]
