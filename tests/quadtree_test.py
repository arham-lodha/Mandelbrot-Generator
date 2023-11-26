import unittest
import numpy as np

from quadtree import QuadTree


class TestQuadTree(unittest.TestCase):

    def test_init(self):
        quadtree = QuadTree((0, 0), (10, 10))
        self.assertTrue(np.array_equal(quadtree.tl, np.array([0, 0])))
        self.assertTrue(np.array_equal(quadtree.br, np.array([10, 10])))
        self.assertEqual(quadtree.rows, 10)
        self.assertEqual(quadtree.cols, 10)
        self.assertEqual(quadtree.children, [])

    def test_split(self):
        quadtree = QuadTree((0, 0), (10, 10))
        children = quadtree.split()
        self.assertEqual(len(children), 4)

        # Check top-left and bottom-right values of each child
        self.assertTrue(np.array_equal(children[0].tl, np.array([1, 1])))
        self.assertTrue(np.array_equal(children[0].br, np.array([5, 5])))

        self.assertTrue(np.array_equal(children[1].tl, np.array([5, 1])))
        self.assertTrue(np.array_equal(children[1].br, np.array([9, 5])))

        self.assertTrue(np.array_equal(children[2].tl, np.array([1, 5])))
        self.assertTrue(np.array_equal(children[2].br, np.array([5, 9])))

        self.assertTrue(np.array_equal(children[3].tl, np.array([5, 5])))
        self.assertTrue(np.array_equal(children[3].br, np.array([9, 9])))

    def test_fill_array(self):
        array = np.zeros((20, 20), dtype=int)
        quadtree = QuadTree((5, 5), (15, 15))
        filled_array = quadtree.fill_array(array, 1)
        self.assertEqual(np.sum(filled_array), 64)

    def test_fill_array_without_boundary(self):
        array = np.zeros((20, 20), dtype=int)
        quadtree = QuadTree((5, 5), (15, 15))
        filled_array = quadtree.fill_array(array, 1, boundary=0)
        self.assertEqual(np.sum(filled_array), 100)

    def test_fill_array_with_small_array(self):
        array = np.zeros((5, 5), dtype=int)
        quadtree = QuadTree((0, 0), (10, 10))
        with self.assertRaises(Exception):
            quadtree.fill_array(array, 1)


if __name__ == '__main__':
    unittest.main()
