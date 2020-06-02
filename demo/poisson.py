import numpy
from scipy import sparse
from scipy.special.orthogonal import ps_roots

from simple_fem import Mesh, FunctionSpace, Q1Element, plot
from simple_fem.assemble import assemble_vector


mesh = Mesh(10, 10)
element = Q1Element()
Q = FunctionSpace(mesh, element)
b = assemble_vector(Q, lambda x: x[0] + x[1])
