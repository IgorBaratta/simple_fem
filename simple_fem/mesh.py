import numpy


class Mesh(object):
    """
    Build a quadrilateral structured unit square mesh.
    """

    def __init__(self, nx: int, ny: int):
        """
        Parameters
        ----------
        nx
            The number of cells in the x direction.
        ny
            The number of cells in the y direction.
        """

        # Only two dimensional quadrilateral meshes supported
        self.reference_cell = ReferenceQuadrilateral()
        self.num_vertices = (nx + 1) * (ny + 1)
        self.num_cells = nx * ny

        # Create meshgrid and use matrix indexing 'ij' to
        # simplify topology computation and dofmap
        x = numpy.linspace(0, 1, nx + 1)
        y = numpy.linspace(0, 1, ny + 1)
        grid = numpy.array(numpy.meshgrid(x, y, indexing="ij")).transpose()

        # Compute coordinate of all nodes in the mesh
        self.vertices = grid.reshape((nx + 1) * (ny + 1), 2)

        # Compute cells - cell-vertex connections
        self.cells = numpy.zeros((self.num_cells, 4), dtype=numpy.int)
        self._topology_computation(nx)

        # Store nx and ny to simplify plotting
        self.nx = nx
        self.ny = ny

    def _topology_computation(self, nx: int):
        """
        Compute cell-vertex connections connections.
        """
        for cell in range(self.num_cells):
            line = cell // nx
            rem = cell % nx
            self.cells[cell] = [
                line * (nx + 1) + rem,
                line * (nx + 1) + rem + 1,
                (line + 1) * (nx + 1) + rem,
                (line + 1) * (nx + 1) + rem + 1,
            ]

    def jacobian(self, i):
        """
        Return the jacobian matrix for cell i.
        Note: since the mesh is structured the jacobian matrix
        is a constant and computed just once and cached.
        """
        local_vert = self.vertices[self.cells[i]]
        dx = local_vert[1, 0] - local_vert[0, 0]
        dy = local_vert[2, 1] - local_vert[0, 1]

        jacobian = numpy.array([[dx, 0], [0, dy]])
        return jacobian

    def area(self, i: int):
        """
        Return the area of cell i.
        Note: since the mesh is structured all elements have the same area.
        """
        return numpy.linalg.det(self.jacobian(i))


class ReferenceQuadrilateral:
    """
    Reference quadrilateral with defined vertices and topology.
    """

    def __init__(self):
        self.dim = 2
        self.num_vertices = 4
        self.num_facets = 4
        self.coordinates = numpy.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
        self.topology = numpy.array([0, 1, 2, 3])
