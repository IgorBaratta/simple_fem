import numpy
from scipy import sparse
from scipy.special.orthogonal import ps_roots

from scipy.sparse.linalg import norm, spsolve
from simple_fem import Mesh, FunctionSpace, Q1Element, plot
from simple_fem.assemble import assemble_matrix, assemble_vector, apply_bc


def test_bc_all():
    mesh = Mesh(1, 1)
    element = Q1Element()
    Q = FunctionSpace(mesh, element)

    A = assemble_matrix(Q, matrix_type="stiffness")
    b = assemble_vector(Q, lambda x: 1)

    dofs = Q.locate_boundary_dofs()
    A = A.tocsr()
    apply_bc(A, b, dofs, value=1)
    x = spsolve(A, b)

    assert numpy.isclose(numpy.sum(x-1), 0)

