from typing import Callable

import numpy
from scipy import integrate, sparse
from scipy.special.orthogonal import ps_roots
from simple_fem.fem import DofMap, Q1Element
from simple_fem.function_space import FunctionSpace
from simple_fem.mesh import Mesh


def assemble_vector(V: FunctionSpace, f: Callable, degree: int = 2) -> numpy.ndarray:
    b = numpy.zeros(V.dofmap.size)
    mesh = V.mesh
    dofmap = V.dofmap

    for i in range(mesh.num_cells):
        local_dofs = dofmap.cell_dofs(i)
        local_coords = mesh.vertices[local_dofs]
        cell_area = mesh.area(i)

        b[local_dofs] += linear_kernel(
            local_coords, f, dofmap.element, cell_area, degree
        )
    return b


def linear_kernel(
    coord: numpy.ndarray, f: Callable, element: Q1Element, area: float, degree: int
) -> numpy.ndarray:

    basis_func = element.basis

    # get quadrature weights and sample points on interval [0, 1]
    # and use tensor product to get 2d sample points
    x, w = ps_roots(degree)
    quad_size = degree * degree
    quad_points = (
        numpy.array(numpy.meshgrid(x, x, indexing="ij"))
        .transpose()
        .reshape(quad_size, 2)
    )
    quad_weights = numpy.outer(w, w).reshape((quad_size, 1))

    # sample input function
    sample_func = numpy.apply_along_axis(f, 1, quad_points).reshape((quad_size, 1))
    sample_basis = numpy.apply_along_axis(basis_func, 1, quad_points)
    result = numpy.sum(quad_weights * sample_func * sample_basis, axis=0) * area

    return result


# def assemble_matrix():
#     pass


# def assemble_cells(data: numpy.ndarray, dofmap: DofMap):
#     mesh = dofmap.mesh
#     local_mat = numpy.zeros(
#         (dofmap.element.num_dofs, dofmap.element.num_dofs), dtype=data.dtype
#     )

#     for idx in range(mesh.num_cells):
#         local_mat.fill(2.0)

#         data[
#             idx * local_mat.size : idx * local_mat.size + local_mat.size
#         ] += local_mat.ravel()


# def sparsity_pattern(dofmap: DofMap):
#     """
#     Returns local COO sparsity pattern.
#     By default when converting to CSR or CSC format,
#     duplicate (i,j) entries are summed together.
#     """
#     num_cells = dofmap.mesh.num_cells
#     num_cell_dofs = dofmap.element.num_dofs

#     rows = numpy.repeat(dofmap.dof_array, num_cell_dofs)
#     cols = numpy.tile(
#         numpy.reshape(dofmap.dof_array, (num_cells, num_cell_dofs)), num_cell_dofs
#     )
#     return rows, cols.ravel()
