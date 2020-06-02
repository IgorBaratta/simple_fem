import numpy

from simple_fem.mesh import Mesh, ReferenceQuadrilateral


class Q1Element:
    def __init__(self):
        self.num_dofs = 4
        self.reference_cell = ReferenceQuadrilateral()

    @property
    def dof_coordinates(self):
        return self.reference_cell.vertices


class DofMap:
    def __init__(self, mesh: Mesh, element=Q1Element()):
        self.mesh = mesh
        self.element = element
        assert mesh.reference_cell == element.reference_cell
        self.dof_array = mesh.cells.ravel()
        self.size = numpy.max(self.dof_array) + 1

    def cell_dofs(self, cell_index: int):
        ndofs = self.element.num_dofs
        return self.dof_array[cell_index*ndofs:(cell_index+1)*ndofs + 1]


if __name__ == '__main__':
    simple_mesh = Mesh(5, 5)
    dofmap = DofMap(simple_mesh)

