import numpy

from simple_fem.mesh import Mesh
from simple_fem.fem import Q1Element, DofMap


class FunctionSpace:
    def __init__(self, mesh: Mesh, element: Q1Element):
        self.mesh = mesh
        self.element = element
        self.dofmap = DofMap(self.mesh, self.element)
    
    



# Quadrilateral elements are particularly amenable to quadrature because integration rules
# can be constructed by taking tensor products of the standard one-dimensional
# Gauss rules.
if __name__ == '__main__':
    simple_mesh = Mesh(10, 10)
    element = Q1Element()
    Q = FunctionSpace(simple_mesh, element)