import numpy
from scipy.sparse.linalg import spsolve

from simple_fem import Mesh, FunctionSpace, Q1Element, plot
from simple_fem.assemble import assemble_matrix, assemble_vector, apply_bc

mesh = Mesh(22, 21)
element = Q1Element()
Q = FunctionSpace(mesh, element)

f = lambda x: 4 * (-x[1] ** 2 + x[1]) * numpy.sin(numpy.pi * x[0])

A = assemble_matrix(Q, matrix_type="stiffness")
b = assemble_vector(Q, f)

dofs = Q.locate_boundary_dofs()
apply_bc(A, b, dofs, value=0)

x = spsolve(A, b)

plot(mesh, x)
