import time
from collections import deque

import numpy as np
from PIL import Image, ImageDraw

from .coloring import color_scheme, generate_colormap_coloring
from quadtree import QuadTree
from .util import calculate_quadtree, calculated_mixed_raster_quadtree, row_raster

import matplotlib.cm


class Mandelbrot:
    def __init__(self,
                 size: tuple[int, int],
                 center=np.array([-0.75, 0]),
                 width: float = 3.5,
                 max_iterations: int = 500,
                 escape_radius=2.0,
                 color=color_scheme[0],
                 smooth=False,
                 raster=False,
                 mixed_raster=False,
                 show_quadtree=False,):
        """
        Mandelbrot Set Generator.

        Parameters:
        - size (tuple[int, int]): The size of the image (width, height).
        - center (np.ndarray): The center coordinates of the Mandelbrot set.
        - width (float): The width of the Mandelbrot set view.
        - max_iterations (int): The maximum number of iterations for set calculations.
        - escape_radius (float): The escape radius for determining if a point is in the Mandelbrot set.
        - color: The color scheme for rendering the Mandelbrot set. Default is color_scheme[0].
        - smooth (bool): Whether to use smooth coloring.
        - period_checking (bool): Whether to perform periodicity checking.
        - raster (bool): Use simple raster approach for rendering (no quadtree).
        - mixed_raster (bool): Use a mixed raster and quadtree approach for rendering.
        - show_quadtree (bool): Whether to visualize the quadtree structure.
        """

        self.size = size
        self.max_iterations = max_iterations
        self.smooth = False

        scale = width / size[0]
        height = scale * size[1]
        offset = center + np.array([-width, height]) / 2

        self.quad_tree = QuadTree(np.zeros(2, dtype=np.int32), np.array([size[0], size[1]], dtype=np.int32))

        self.x = np.linspace(0, size[0], num=size[0], dtype=np.float64) * scale + offset[0]
        self.y = np.linspace(0, size[1], num=size[1], dtype=np.float64) * -scale + offset[1]

        self.y = np.abs(self.y)

        # self.x, self.y = np.meshgrid(self.x, self.y)

        self.pixels = np.zeros((size[1], size[0], 3), dtype=np.uint8) if not mixed_raster or not raster else np.ones(
            (size[1], size[0], 3), dtype=np.uint8) * -1

        self.color_scheme, self.color_mode = color
        self.smooth = smooth

        self.escape_radius = escape_radius
        self.escape_radius_squared = escape_radius * escape_radius

        self.raster = raster
        self.mixed_raster = mixed_raster
        self.period_checking = True

        self.show_quadtree = not self.raster and show_quadtree

        self.num_computed = 0
        self.percent_completed = 0.0
        self.width = width

    def generate(self):
        """
        Generates the Mandelbrot set based on the configured parameters.
        """
        if self.raster:
            memo = {}
            for i in range(self.size[1]):
                row_raster(self.pixels, i, self.x, self.y, self.max_iterations,
                           self.escape_radius,
                           self.smooth,
                           self.color_scheme,
                           self.num_computed,
                           self.period_checking, memo)
        elif self.mixed_raster:
            memo = dict()
            self.quad_tree.split(boundary=0)
            queue: deque[QuadTree] = deque(self.quad_tree.children)

            while queue:
                quad_tree = queue.popleft()

                split, border = calculated_mixed_raster_quadtree(quad_tree, self.pixels, self.x, self.y,
                                                                 self.max_iterations,
                                                                 self.escape_radius,
                                                                 self.smooth,
                                                                 self.color_scheme,
                                                                 self.num_computed,
                                                                 self.period_checking, memo)

                if split:
                    quad_tree.split()
                    for child in quad_tree.children:
                        queue.append(child)
                else:
                    quad_tree.fill_array(self.pixels, border)

            for i in range(self.size[1]):
                row_raster(self.pixels, i, self.x, self.y, self.max_iterations,
                           self.escape_radius,
                           self.smooth,
                           self.color_scheme,
                           self.num_computed,
                           self.period_checking, memo)
        else:
            memo = {}
            self.quad_tree.split(boundary=0)
            queue: deque[QuadTree] = deque(self.quad_tree.children)

            while queue:
                quad_tree = queue.popleft()

                split, border = calculate_quadtree(quad_tree, self.pixels, self.x, self.y, self.max_iterations,
                                                   self.escape_radius,
                                                   self.smooth,
                                                   self.color_scheme,
                                                   self.num_computed,
                                                   self.period_checking, memo)

                if split:
                    quad_tree.split()
                    for child in quad_tree.children:
                        queue.append(child)
                else:
                    quad_tree.fill_array(self.pixels, border)

    def render(self, filename="image.png"):
        """
        Renders and saves the generated Mandelbrot set image.

        Parameters:
        - filename (str): The filename to save the image.
        """

        img = Image.fromarray(self.pixels, mode=self.color_mode).convert('RGB')

        if self.show_quadtree:
            self.draw_quadtree(img)

        img.save(filename)

    def draw_quadtree(self, img):
        """
        Draws the quadtree boundaries on the image.

        Parameters:
        - img (Image): The image object to draw on.
        """
        draw = ImageDraw.Draw(img)
        queue: deque[QuadTree] = deque([self.quad_tree])

        while queue:
            quad_tree = queue.popleft()
            draw.rectangle(((quad_tree.tl[0], quad_tree.tl[1]), (quad_tree.br[0], quad_tree.br[1])), outline="red")

            for child in quad_tree.children:
                queue.append(child)