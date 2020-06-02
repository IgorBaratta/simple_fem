import numpy
from scipy import sparse, integrate

from simple_fem import Mesh, FunctionSpace, Q1Element
from simple_fem.assemble import assemble_vector


mesh = Mesh(2, 2)
element = Q1Element()
V = FunctionSpace(mesh, element)

f = lambda x: 1
b = assemble_vector(V, f)
