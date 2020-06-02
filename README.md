# simple_fem
![CI](https://github.com/IgorBaratta/simple_fem/workflows/Python%20package/badge.svg)

## Usage:

```python3
from scipy.sparse.linalg import spsolve
from simple_fem import Mesh, FunctionsSpace, Q1Element, plot

mesh = Mesh(10, 10)
e = Q1Element()
V = FunctionSpace(mesh, e)

f = lambda x : 4*(-x[1]**2 + x[1])*numpy.sin(numpi.pi*x[0])

A = assemble_matrix(V, "poisson")
b = assemble_vector(V, f)

dofs = locate_boundary_dofs(V)
apply_bc(A, b, dofs, value=0)

x  = spsolve(A, b)

plot(V, x)

```
