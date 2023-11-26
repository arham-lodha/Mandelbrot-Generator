import numpy as np


class QuadTree:
    def __init__(self, top_left, bottom_right):
        """
        Initializes a QuadTree node with the specified top-left and bottom-right coordinates.

        Parameters:
        top_left (tuple[int, int]): Top-left coordinates of the bounding box.
        bottom_right (tuple[int, int]): Bottom-right coordinates of the bounding box.
        """
        self.tl = top_left
        self.br = bottom_right

        self.rows: int = self.br[1] - self.tl[1]
        self.cols: int = self.br[0] - self.tl[0]

        self.children: list[QuadTree] = []

    def split(self, boundary=1):
        """
        Splits the current QuadTree node into four child nodes.

        Parameters:
        boundary (int): Optional boundary to leave empty around each child node.

        Returns:
        list[QuadTree]: List of four child QuadTree nodes after splitting.

        Raises:
        Exception: If the node cannot be split.
        """
        tl = np.array([self.tl[0] + boundary, self.tl[1] + boundary], dtype=np.int32)
        br = np.array([self.br[0] - boundary, self.br[1] - boundary], dtype=np.int32)

        cols = br[0] - tl[0]
        rows = br[1] - tl[1]

        if rows < 1 or cols < 1:
            raise Exception("Cannot split node.")

        if rows == 1 and cols == 1:
            self.children = [
                QuadTree(tl, br)
            ]
            return self.children

        smaller_box_size = (cols // 2, rows // 2)

        if rows == 1:
            self.children = [
                QuadTree(tl, np.array([tl[0] + smaller_box_size[0], br[1]], dtype=np.int32)),
                QuadTree(np.array([tl[0] + smaller_box_size[0], tl[1]], dtype=np.int32), br)
            ]
            return self.children

        if cols == 1:
            self.children = [
                QuadTree(tl, np.array([br[0], tl[1] + smaller_box_size[1]], dtype=np.int32)),
                QuadTree(np.array([tl[0], tl[1] + smaller_box_size[1]], dtype=np.int32), br)
            ]
            return self.children

        self.children = [
            QuadTree(
                tl, np.array([tl[0] + smaller_box_size[0], tl[1] + smaller_box_size[1]], dtype=np.int32)
            ),
            QuadTree(
                np.array([tl[0] + smaller_box_size[0], tl[1]], dtype=np.int32),
                np.array([br[0], tl[1] + smaller_box_size[1]], dtype=np.int32)
            ),
            QuadTree(
                np.array([tl[0], tl[1] + smaller_box_size[1]], dtype=np.int32),
                np.array([tl[0] + smaller_box_size[0], br[1]], dtype=np.int32)
            ),
            QuadTree(
                np.array([tl[0] + smaller_box_size[0], tl[1] + smaller_box_size[1]], dtype=np.int32),
                br
            )
        ]

        return self.children

    def fill_array(self, array, value, boundary=1):
        """
        Fills a portion of a 2D array with a specified value based on the current QuadTree node.

        Parameters:
        array (numpy.ndarray): 2D array to fill.
        value: Value to fill in the array.
        boundary (int): Optional boundary to leave empty around the filled area.

        Returns:
        numpy.ndarray: Updated array after filling.

        Raises:
        Exception: If the input array is too small.
        """
        if len(array) < self.br[1] and len(array[0]) < self.br[0]:
            raise Exception("Array too small")

        array[self.tl[1] + boundary: self.br[1] - boundary, self.tl[0] + boundary: self.br[0] - boundary] = value

        return array

    def __str__(self) -> str:
        """
        Returns a string representation of the QuadTree node.

        Returns:
        str: String representation of the QuadTree node.
        """
        return f"{self.tl} {self.br}"
