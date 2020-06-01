import numpy


class Mesh(object):
    """
    Build a structured unit square mesh.


    """

    def __init__(self, nx: int, ny: int):
        """
        :param nx: The number of cells in the x direction.
        :param ny: The number of cells in the y direction.
        """

        # Only two dimensional meshes are currently supported
        self.dim = 2
        self.num_vertices = (nx + 1) * (ny + 1)
        self.num_cells = nx * ny

        x = numpy.linspace(0, 1, nx + 1)
        y = numpy.linspace(0, 1, ny + 1)

        # use matrix indexing ij to simplify topology computation
        grid = numpy.array(numpy.meshgrid(x, y, indexing='ij')).transpose()

        self.vertices = grid.reshape((nx + 1) * (ny + 1), 2)
        self.cells = numpy.zeros((nx * ny, 4), dtype=numpy.int32)
        self._topology_computation(nx, ny)

    def _topology_computation(self, nx, ny):
        for cell in range(self.num_cells):
            line = cell // nx
            rem = cell % nx
            self.cells[cell] = [line * (nx + 1) + rem,
                                line * (nx + 1) + rem + 1,
                                (line + 1) * (nx + 1) + rem,
                                (line + 1) * (nx + 1) + rem + 1]

    def jacobian(self, i):
        """
        Return the Jacobian matrix the ith cell.
        """
        raise NotImplementedError


class ReferenceCell:
    """
    Reference quadrilateral with defined vertices
    and topology:

    """
    def __init__(self):
        self.vertices = [[0.0, 0.0],
                         [0.0, 1.0],
                         [1.0, 0.0],
                         [1.0, 1.1]]
        self.topology = [0, 1, 2, 3]
