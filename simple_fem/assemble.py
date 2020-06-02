import numpy
from scipy import sparse

from simple_fem.mesh import Mesh
from simple_fem.fem import DofMap

def assemble_matrix():
    pass


def assemble_cells(data: numpy.ndarray, dofmap: DofMap):
    mesh = dofmap.mesh
    local_mat = numpy.zeros((dofmap.element.num_dofs, 
                             dofmap.element.num_dofs), dtype=data.dtype)
    
    for idx in range(mesh.num_cells):
        local_mat.fill(2.0)
        
        data[idx * local_mat.size : idx * local_mat.size + local_mat.size] += local_mat.ravel()


def sparsity_pattern(dofmap: DofMap):
    """
    Returns local COO sparsity pattern.
    By default when converting to CSR or CSC format, 
    duplicate (i,j) entries are summed together. 
    """
    num_cells = dofmap.mesh.num_cells
    num_cell_dofs = dofmap.element.num_dofs
    
    rows = numpy.repeat(dofmap.dof_array, num_cell_dofs)
    cols = numpy.tile(numpy.reshape(dofmap.dof_array, (num_cells, num_cell_dofs)), num_cell_dofs)
    return rows, cols.ravel()