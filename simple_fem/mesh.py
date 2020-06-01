import numpy


class Mesh(object):
    """
    Build a quadrilateral structured unit square mesh.
    """

    def __init__(self, nx: int, ny: int):
        """
        :param nx: The number of cells in the x direction.
        :param ny: The number of cells in the y direction.
        """

        # Only two dimensional quadrilateral meshes supported
        self.reference_cell = ReferenceQuadrilateral()
        self.num_vertices = (nx + 1) * (ny + 1)
        self.num_cells = nx * ny

        # Create meshgrid and use matrix indexing 'ij' to
        # simplify topology computation
        x = numpy.linspace(0, 1, nx + 1)
        y = numpy.linspace(0, 1, ny + 1)
        grid = numpy.array(numpy.meshgrid(x, y, indexing='ij')).transpose()

        self.vertices = grid.reshape((nx + 1) * (ny + 1), 2)
        self.cells = numpy.zeros((self.num_cells, 4), dtype=numpy.int32)
        self._topology_computation(nx)

    def _topology_computation(self, nx):
        for cell in range(self.num_cells):
            line = cell // nx
            rem = cell % nx
            self.cells[cell] = [line * (nx + 1) + rem,
                                line * (nx + 1) + rem + 1,
                                (line + 1) * (nx + 1) + rem,
                                (line + 1) * (nx + 1) + rem + 1]

    def jacobian(self, i: int):
        """
        Return the Jacobian matrix the ith cell.
        """
        raise NotImplementedError


class ReferenceQuadrilateral:
    """
    Reference quadrilateral with defined vertices
    and topology.

    Font: Logg, Anders, Kent-Andre Mardal, and Garth Wells, eds. Automated solution of
    differential equations by the finite element method: The FEniCS book. Vol. 84.
    Springer Science & Business Media, 2012. -  Page
    """

    def __init__(self):
        self.dim = 2
        self.num_vertices = 4
        self.num_facets = 4
        self.vertices = numpy.array([[0.0, 0.0], [0.0, 1.0],
                                     [1.0, 0.0], [1.0, 1.1]])
        self.topology = numpy.array([0, 1, 2, 3])
