import numpy
from scipy import sparse
from scipy.special.orthogonal import ps_roots

from scipy.sparse.linalg import norm
from simple_fem import Mesh, FunctionSpace, Q1Element, plot
from simple_fem.assemble import assemble_matrix


mesh = Mesh(10, 10)
element = Q1Element()
Q = FunctionSpace(mesh, element)
A = assemble_matrix(Q, matrix_type="stiffness")