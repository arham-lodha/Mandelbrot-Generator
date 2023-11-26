from collections import deque

import numpy as np
from PIL import Image, ImageDraw
from numba import u8, u1

from quadtree import QuadTree
from .normal_quadtree import calculate_quadtree, compute_fast_quadtree
from .mixed_quadtree import calculate_mixed, fast_mixed_quadtree
from .raster import compute_raster


class AcceleratedMandelbrot:
    def __init__(self,
                 size: tuple[int, int],
                 center=np.array([-0.75, 0]),
                 width: float = 3.5,
                 max_iterations: int = 500,
                 escape_radius=2.0,
                 color_scheme: int = 0,
                 color_map=np.array([[]], dtype=np.uint8),
                 smooth=False,
                 raster = False,
                 mixed_raster=False,
                 fast_quadtree = True,
                 show_quadtree=False,
                 ):

        """
        Initializes the AcceleratedMandelbrot generator with the specified parameters.

        Parameters:
        - size (tuple[int, int]): Size of the output image in pixels (width, height).
        - center (numpy.ndarray): Center coordinates of the fractal in the complex plane.
        - width (float): Width of the fractal region in the complex plane.
        - max_iterations (int): Maximum number of iterations to determine the fractal.
        - escape_radius (float): Radius for determining escape condition.
        - color_scheme (int): Color scheme identifier.
        - color_map (numpy.ndarray): Color map for mapping fractal values to colors.
        - smooth (bool): Enable smooth coloring.
        - raster (bool): Use raster mode for generating the fractal.
        - mixed_raster (bool): Use mixed raster mode.
        - fast_quadtree (bool): Use fast quadtree mode.
        - show_quadtree (bool): Show quadtree boundaries in the output image.
        """

        self.size = size
        self.max_iterations = u8.cast_python_value(max_iterations)
        self.smooth = smooth

        self.raster = raster
        self.mixed_raster = mixed_raster
        self.period_checking = True
        self.fast_quadtree = fast_quadtree if not self.raster else False

        scale = width / size[0]
        height = scale * size[1]
        offset = center + np.array([-width, height]) / 2

        self.quad_tree = QuadTree(np.zeros(2, dtype=np.int32), np.array([size[0], size[1]], dtype=np.int32))
        self.x = np.linspace(0, size[0], num=size[0], dtype=np.float64) * scale + offset[0]
        self.y = np.linspace(0, size[1], num=size[1], dtype=np.float64) * -scale + offset[1]

        self.y = np.abs(self.y)

        self.pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8)
        self.color_scheme = u1.cast_python_value(color_scheme)
        self.color_mode = "RGB"
        self.color_map = color_map if self.color_scheme == 3 else np.array([[]], dtype=np.uint8)

        if self.color_scheme == 1:
            self.color_mode = "HSV"
        elif self.color_scheme == 2:
            self.color_mode = "HSV"

        self.smooth = smooth

        self.escape_radius = escape_radius
        self.escape_radius_squared = escape_radius * escape_radius

        self.show_quadtree = not self.raster and show_quadtree

        self.num_computed = 0
        self.percent_completed = 0.0
        self.width = width

    def generate(self):
        """
        Generates the Mandelbrot fractal based on the specified configuration.
        """
        if self.raster:
            compute_raster(self.pixels, np.zeros(shape=(1, 0), dtype=bool), self.x, self.y, self.max_iterations,
                           self.escape_radius,
                           self.smooth,
                           self.color_scheme,
                           self.color_map,
                           self.period_checking)
        if self.mixed_raster:
            if self.fast_quadtree:
                self.quad_tree.split(boundary=0)
                queue: deque[QuadTree] = deque(self.quad_tree.children)
                interval = np.array([[quad.tl, quad.br] for quad in self.quad_tree.children])

                seen = np.zeros(shape=(self.size[1], self.size[0]), dtype=bool)

                while queue:
                    result_list = fast_mixed_quadtree(interval, self.pixels, seen,
                                                      self.x,
                                                      self.y,
                                                      self.max_iterations,
                                                      self.escape_radius,
                                                      self.smooth,
                                                      self.color_scheme,
                                                      self.color_map,
                                                      self.period_checking)

                    temp_interval = []

                    for i, result in enumerate(result_list):
                        quadtree = queue.popleft()
                        split = result[0] != 0
                        isMandelbrot = result[1] != 0
                        border = result[2:]

                        if split:
                            quadtree.split()
                            queue.extend(quadtree.children)
                            temp_interval += [[quad.tl, quad.br] for quad in quadtree.children]
                        elif isMandelbrot:
                            quadtree.fill_array(self.pixels, border)
                            quadtree.fill_array(seen, True)

                    interval = np.array(temp_interval)

                compute_raster(self.pixels, seen, self.x, self.y, self.max_iterations, self.escape_radius,
                               self.smooth,
                               self.color_scheme,
                               self.color_map,
                               self.period_checking)
            else:
                self.mixed_quadtree()
        else:
            if self.fast_quadtree:
                self.quad_tree.split(boundary=0)
                queue: deque[QuadTree] = deque(self.quad_tree.children)
                interval = np.array([[quad.tl, quad.br] for quad in self.quad_tree.children])

                while queue:
                    result_list = compute_fast_quadtree(interval, self.pixels,
                                                        self.x,
                                                        self.y,
                                                        self.max_iterations,
                                                        self.escape_radius,
                                                        self.smooth,
                                                        self.color_scheme,
                                                        self.color_map,
                                                        self.period_checking)

                    temp_interval = []

                    for i, result in enumerate(result_list):
                        quadtree = queue.popleft()
                        split = result[0] != 0
                        border = result[1:]

                        if split:
                            quadtree.split()
                            queue.extend(quadtree.children)
                            temp_interval += [[quad.tl, quad.br] for quad in quadtree.children]
                        else:
                            quadtree.fill_array(self.pixels, border)

                    interval = np.array(temp_interval)
            else:
                self.normal_quadtree()

    def normal_quadtree(self):
        """
        Generates the Mandelbrot fractal using normal quadtree computation.
        """
        self.quad_tree.split(boundary=0)
        queue: deque[QuadTree] = deque(self.quad_tree.children)

        while queue:
            quad_tree = queue.popleft()

            result = calculate_quadtree(quad_tree.tl, quad_tree.br, self.pixels,
                                        self.x,
                                        self.y,
                                        self.max_iterations,
                                        self.escape_radius,
                                        self.smooth,
                                        self.color_scheme,
                                        self.color_map,
                                        self.period_checking)

            split = result[0] != 0
            border = result[1:]

            if split:
                quad_tree.split()
                queue.extend(quad_tree.children)
            else:
                quad_tree.fill_array(self.pixels, border)

    def mixed_quadtree(self):
        """
        Generates the Mandelbrot fractal using mixed quadtree computation.
        """

        self.quad_tree.split(boundary=0)
        queue: deque[QuadTree] = deque(self.quad_tree.children)

        seen = np.zeros(shape=(self.size[1], self.size[0]), dtype=bool)

        while queue:
            quad_tree = queue.popleft()

            result = calculate_mixed(quad_tree.tl, quad_tree.br, self.pixels, seen,
                                     self.x,
                                     self.y,
                                     self.max_iterations,
                                     self.escape_radius,
                                     self.smooth,
                                     self.color_scheme,
                                     self.color_map,
                                     self.period_checking)

            split = result[0] != 0
            isMandelbrot = result[1] != 0
            border = result[2:]

            if split:
                quad_tree.split()
                queue.extend(quad_tree.children)
            elif isMandelbrot:
                quad_tree.fill_array(self.pixels, border)
                quad_tree.fill_array(seen, True)

        compute_raster(self.pixels, seen, self.x, self.y, self.max_iterations, self.escape_radius,
                       self.smooth,
                       self.color_scheme,
                       self.color_map,
                       self.period_checking)

    def render(self, filename="image.png"):
        """
        Renders and saves the generated Mandelbrot fractal as an image.

        Parameters:
        - filename (str): Output file name.
        """
        img = Image.fromarray(self.pixels, mode=self.color_mode).convert('RGB')

        if self.show_quadtree:
            self.draw_quadtree(img)

        img.save(filename)

    def draw_quadtree(self, img):
        """
        Draws quadtree boundaries on the output image.

        Parameters:
        - img (Image): Image object to draw quadtree boundaries on.
        """

        draw = ImageDraw.Draw(img)
        queue: deque[QuadTree] = deque([self.quad_tree])

        while queue:
            quad_tree = queue.popleft()
            draw.rectangle(((quad_tree.tl[0], quad_tree.tl[1]), (quad_tree.br[0], quad_tree.br[1])), outline="red")

            for child in quad_tree.children:
                queue.append(child)
