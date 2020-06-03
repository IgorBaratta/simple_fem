import numpy
from scipy import sparse
from scipy.sparse.linalg import norm

from simple_fem import Mesh, FunctionSpace, Q1Element
from simple_fem.assemble import assemble_vector, assemble_matrix


def test_Q1Element():
    element = Q1Element()
    dof_coord = element.dof_coordinates
    for i in range(element.num_dofs):
        delta = numpy.zeros(element.num_dofs)
        delta[i] = 1.
        assert numpy.allclose(element.basis(dof_coord[i]), delta)

def test_Q1Element_derivative():
    element = Q1Element()
    dx = element.basis_derivative[0]
    dy = element.basis_derivative[1]

    assert (dx([0,0]) == [-1, 1, 0, 0]).all()
    assert (dy([0,0]) == [-1, 0, 1, 0]).all()

    assert (dx([1,1]) == [0, 0, -1, 1]).all()
    assert (dy([1,1]) == [0, -1, 0, 1]).all()



def test_assemble_single_cell():
    mesh = Mesh(1, 1)
    element = Q1Element()
    V = FunctionSpace(mesh, element)
    b = assemble_vector(V, lambda x: 1)
    assert numpy.allclose(b, [0.25, 0.25, 0.25, 0.25])


def test_assemble_vector():
    mesh = Mesh(10, 10)
    element = Q1Element()
    Q = FunctionSpace(mesh, element)
    b = assemble_vector(Q, lambda x: x[0] + x[1])
    assert numpy.isclose(numpy.sum(b), 1.)


def test_assemble_mass_matrix():
    mesh = Mesh(2, 2)
    element = Q1Element()
    Q = FunctionSpace(mesh, element)
    A = assemble_matrix(Q, "mass", degree=5)
    assert numpy.isclose(norm(A), 0.194445, 1e-05)


def test_assemble_stiffness_matrix():
    mesh = Mesh(1, 1)
    element = Q1Element()
    Q = FunctionSpace(mesh, element)
    A = assemble_matrix(Q, "stiffness", degree=5)
    assert numpy.isclose(norm(A), 1.563472, 1e-05)
