from typing import Callable

import numpy
from scipy import sparse
from scipy.special.orthogonal import ps_roots
from simple_fem.fem import DofMap, Q1Element
from simple_fem.function_space import FunctionSpace
from simple_fem.mesh import Mesh
from simple_fem.quadrature import Quadrature


def assemble_vector(V: FunctionSpace, f: Callable, degree: int = 2) -> numpy.ndarray:
    b = numpy.zeros(V.dofmap.size)
    mesh = V.mesh
    dofmap = V.dofmap

    quad = Quadrature(degree)

    for i in range(mesh.num_cells):
        local_dofs = dofmap.cell_dofs(i)
        local_coords = mesh.vertices[local_dofs]
        cell_area = mesh.area(i)

        b[local_dofs] += linear_kernel(
            local_coords, f, dofmap.element, cell_area, quad
        )
    return b


def linear_kernel(
    coord: numpy.ndarray, f: Callable,
    element: Q1Element, area: float, quad: Quadrature
) -> numpy.ndarray:

    # Sample basis functions on quadrature points
    sample_basis = numpy.apply_along_axis(element.basis, 1, quad.points)

    # sample input function on quadrature points mapped back to the physical domain
    mapped_points = numpy.dot(sample_basis, coord)
    sample_func = numpy.apply_along_axis(
        f, 1, mapped_points).reshape((quad.size, 1))

    # Compute integral result
    result = numpy.sum(quad.weights * sample_func *
                       sample_basis, axis=0) * area
    return result


def assemble_matrix(V: FunctionSpace, matrix_type: str = "mass", degree: int = 4):
    mesh = V.mesh
    dofmap = V.dofmap
    element = V.dofmap.element
    Ae = numpy.zeros((element.num_dofs, element.num_dofs))
    data = numpy.zeros(mesh.num_cells * element.num_dofs * element.num_dofs)

    quad = Quadrature(degree)

    if matrix_type.lower() == "mass":
        kernel = mass_kernel
    elif matrix_type.lower() == "stiffness":
        kernel = stiffness_kernel
    else:
        raise NotImplementedError

    for i in range(mesh.num_cells):
        Ae.fill(0)
        local_dofs = dofmap.cell_dofs(i)
        local_coords = mesh.vertices[local_dofs]
        cell_area = mesh.area(i)
        Ae[:] = kernel(local_coords, element, cell_area, quad)
        data[i*Ae.size: i * Ae.size + Ae.size] = Ae.ravel()

    A = sparse.coo_matrix((data, sparsity_pattern(dofmap)),
                          shape=(dofmap.size, dofmap.size))
    return A


def mass_kernel(coord: numpy.ndarray, element: Q1Element, area: float, quad: Quadrature) -> numpy.ndarray:

    # Sample basis functions on quadrature points
    sample_basis = numpy.apply_along_axis(element.basis, 1, quad.points)

    # Ae_{i,j} = \sum_{p} phi(i,p) phi(p,j) weights(p)
    Ae = (sample_basis*quad.weights).T @ sample_basis * area
    return Ae


def stiffness_kernel(coord: numpy.ndarray, element: Q1Element, area: float, quad: Quadrature) -> numpy.ndarray:
    # Samble basis derivatives on quadrature points
    sample_dx = numpy.apply_along_axis(
        element.basis_derivative[0], 1, quad.points)
    sample_dy = numpy.apply_along_axis(
        element.basis_derivative[1], 1, quad.points)

    # Add contributions to local matrix componentwise, first dx then dy
    Ae = numpy.zeros((element.num_dofs, element.num_dofs))
    # Ae_{i,j} = \sum_i \sum_{p} d_{x_i} phi(i,p) d_{x_i} phi(p,j) weights(p)
    Ae += (sample_dx*quad.weights).T @ sample_dx * area
    Ae += (sample_dy*quad.weights).T @ sample_dy * area
    return Ae


def sparsity_pattern(dofmap: DofMap):
    """
    Returns local COO sparsity pattern. By default when converting to CSR or CSC format,
    duplicate (i,j) entries are summed together.
    """
    num_cells = dofmap.mesh.num_cells
    dofs_per_cell = dofmap.element.num_dofs

    rows = numpy.repeat(dofmap.dof_array, dofs_per_cell)
    cols = numpy.tile(
        numpy.reshape(dofmap.dof_array, (num_cells,
                                         dofs_per_cell)), dofs_per_cell
    )
    return rows, cols.ravel()


def apply_bc(A: sparse.spmatrix, b: numpy.ndarray, dofs: numpy.ndarray, value: float = 0):
    if sparse.isspmatrix_coo(A):
        for row in dofs:
            A.data[A.row == row] = 0.
            A.data[(A.row == row) * (A.row == row)] = 1
    else:
        raise TypeError("This functions only accepts COO matrices")
    b[dofs] = value
