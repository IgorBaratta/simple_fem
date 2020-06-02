import numpy

from simple_fem.mesh import Mesh, ReferenceQuadrilateral


class Q1Element:
    def __init__(self):
        self.num_dofs = 4
        self.reference_cell = ReferenceQuadrilateral()

        # 2D basis basis functions for quadrilatrals can b constructed as tensor 
        # products of 1D functions
        self.basis = lambda x, y: numpy.outer([1 - x, x], [1 - y, y]).flatten()
        self.basis_derivative = lambda x, y: [numpy.outer([-1, 1], [1 - y, y]).flatten(),
                                              numpy.outer([1 - x, x], [-1, 1]).flatten()]
    @property
    def dof_coordinates(self):
        return self.reference_cell.coordinates


class DofMap:
    def __init__(self, mesh: Mesh, element=Q1Element()):
        self.mesh = mesh
        self.element = element
        self.dof_array = mesh.cells.ravel()
        self.size = numpy.max(self.dof_array) + 1

    def cell_dofs(self, cell_index: int):
        ndofs = self.element.num_dofs
        return self.dof_array[cell_index*ndofs:(cell_index+1)*ndofs + 1]


if __name__ == '__main__':
    simple_mesh = Mesh(10, 10)
    dofmap = DofMap(simple_mesh)

