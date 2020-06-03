import matplotlib.pyplot as plt
import matplotlib.collections
import numpy

from simple_fem import Mesh


def plot(mesh: Mesh, values=None, show_vertices=True):
    """
    Plot 2d function and mesh.
    """
    parameters = {"edgecolor": "k", "cmap": "rainbow", "linewidths": (0.5,)}

    if values is not None:
        if values.size != mesh.num_vertices:
            raise ValueError("dimension mismatch")
    else:
        parameters["facecolor"] = "None"

    _, ax = plt.subplots()
    if show_vertices:
        ax.plot(mesh.vertices[:, 0], mesh.vertices[:, 1],
                marker=".", ls="", color="k")

    if values is not None:
        x1 = numpy.linspace(0, 1, mesh.nx + 1)
        y1 = numpy.linspace(0, 1, mesh.ny + 1)
        grid = numpy.meshgrid(x1, y1)
        plt.contourf(grid[1], grid[0], values.reshape((mesh.nx + 1, mesh.ny + 1)))
        parameters["facecolor"] = "None"
        plt.colorbar()

    pc = add_poly(mesh, **parameters)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")

    plt.show()


def reorder_counterclockwise(cells: numpy.ndarray):
    """
    Reorder cell topology so that V0->V1->V2->V3->V0
    forms a closed path and the resulting quadrilateral
    is convex.
    """
    perm = numpy.array([0, 1, 3, 2], dtype=numpy.int32)
    new_cell_order = cells[:, perm]
    return new_cell_order


def add_poly(mesh, **kwargs):
    ordered_cells = reorder_counterclockwise(mesh.cells)
    verts = mesh.vertices[ordered_cells]
    pc = matplotlib.collections.PolyCollection(verts, **kwargs)
    return pc


if __name__ == "__main__":
    simple_mesh = Mesh(30, 30)
    cell_values = numpy.arange(simple_mesh.num_cells)
    plot(simple_mesh, cell_values, False)
