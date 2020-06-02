import numpy
from scipy import sparse

from simple_fem import Mesh, FunctionSpace, Q1Element
from simple_fem.assemble import assemble_vector


def test_Q1Element():
    element = Q1Element()
    dof_coord = element.dof_coordinates
    for i in range(element.num_dofs):
        delta = numpy.zeros(element.num_dofs)
        delta[i] = 1
        assert numpy.allclose(element.basis(dof_coord[i]), delta)


def test_assemble_vector_single():
    mesh = Mesh(1, 1)
    element = Q1Element()
    V = FunctionSpace(mesh, element)
    def f(x): return 1
    b = assemble_vector(V, f)
    assert numpy.allclose(b, [0.25, 0.25, 0.25, 0.25])
