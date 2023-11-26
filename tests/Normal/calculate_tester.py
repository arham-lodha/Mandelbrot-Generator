import unittest

from Normal.coloring import color_scheme
from Normal.util import in_main_body, calculate


class TestFractalFunctions(unittest.TestCase):

    def test_in_main_body(self):
        self.assertTrue(in_main_body(0.2, 0.3))
        self.assertFalse(in_main_body(1.0, 0.0))

    def test_calculate(self):
        # Example test case, modify as needed
        x0 = 0.2
        y0 = 0.3
        max_iterations = 100
        escape_radius = 2.0
        smooth = True
        period_checking = False

        result = calculate(x0, y0, max_iterations, escape_radius, smooth, color_scheme[0][0],0, period_checking)

        # Assuming the output format is [isMaxIteration, R, G, B]
        self.assertTrue(result[1] == 1)


if __name__ == '__main__':
    unittest.main()
