import numpy
from simple_fem.mesh import Mesh, ReferenceQuadrilateral

def test_single_cell():
    """
    Test mesh order (the same order is used for the dofmap)
    """
    mesh = Mesh(1, 1)
    reference_cell = ReferenceQuadrilateral()
    assert (mesh.vertices[mesh.cells[0]] == reference_cell.coordinates).all()


def test_mesh_dimensions():
    """
    Test some mesh properties
    """
    mesh = Mesh(10, 5)
    assert mesh.cells.shape == (mesh.num_cells, 4)
    assert mesh.num_cells == 10*5
    assert numpy.max(mesh.cells) + 1 == mesh.num_vertices
    assert numpy.isclose(mesh.area(0), mesh.area(1))
    assert numpy.isclose(mesh.area(10), 1/mesh.num_cells)


def plot_mesh():
    pass

