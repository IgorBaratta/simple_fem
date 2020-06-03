import numpy

from simple_fem.mesh import Mesh
from simple_fem.fem import Q1Element, DofMap


class FunctionSpace:
    def __init__(self, mesh: Mesh, element: Q1Element):
        self._mesh = mesh
        self._element = element
        self._dofmap = DofMap(self._mesh, self._element)

    @property
    def dofmap(self):
        """
        Return the dofmap associated with the function space.
        """
        return self._dofmap

    @property
    def mesh(self):
        """
        Return the mesh on which the function space is defined.
        """
        return self._mesh

    @property
    def element(self):
        """
        Return the finite element tha defines the function space.
        """
        return self._element

    def locate_boundary_dofs(self):
        """
        Return indices of the boundary dofs.
        x0==0 or x0==1 or x1==0 or x1==1
        """
        x_boundary = numpy.logical_or(
            numpy.isclose(self._mesh.vertices[:, 0], 0.),
            numpy.isclose(self._mesh.vertices[:, 0], 1.)
        )

        y_boundary = numpy.logical_or(
            numpy.isclose(self._mesh.vertices[:, 1], 0.),
            numpy.isclose(self._mesh.vertices[:, 1], 1.)
        )

        return numpy.where(numpy.logical_or(x_boundary, y_boundary))[0]
