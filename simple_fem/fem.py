import numpy

from simple_fem.mesh import Mesh, ReferenceQuadrilateral


class Q1Element:
    def __init__(self):
        self.num_dofs = 4
        self.reference_cell = ReferenceQuadrilateral()

        # 2D basis basis functions for quadrilaterals can be constructed
        # as tensor products of 1D functions
        self.basis = lambda x: numpy.outer(
            [1 - x[0], x[0]], [1 - x[1], x[1]]).T.flatten()

        self.basis_derivative = [lambda x: numpy.outer([-1, 1], [1 - x[1], x[1]]).T.flatten(),
                                 lambda x: numpy.outer([1 - x[0], x[0]], [-1, 1]).T.flatten()]

    @property
    def dof_coordinates(self):
        return self.reference_cell.coordinates


class DofMap:
    def __init__(self, mesh: Mesh, element=Q1Element()):
        self.mesh = mesh
        self.element = element
        self.dof_array = mesh.cells.ravel()
        self._size = numpy.max(self.dof_array) + 1

    def cell_dofs(self, i: int) -> numpy.ndarray:
        """
        Return ofs (global numbering) for cell i
        """
        ndofs = self.element.num_dofs
        return self.dof_array[i * ndofs: (i + 1) * ndofs]

    @property
    def size(self) -> int:
        """
        Return the number of degrees of freedom.
        """
        return self._size


if __name__ == "__main__":
    simple_mesh = Mesh(10, 10)
    dofmap = DofMap(simple_mesh)
