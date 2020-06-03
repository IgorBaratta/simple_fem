import matplotlib.pyplot as plt
import matplotlib.collections
import numpy

from simple_fem import Mesh


def plot(mesh: Mesh, values=None, show_vertices=True):
    """
    Plot 2d function and mesh.
    If plot nx == ny, plot values using contourf, else plot
    values by cell (mean value of the function within the cell) .
    """
    parameters = {"edgecolor": "k",
                  "linewidths": (0.5,)}

    if values is not None:
        if values.size != mesh.num_vertices:
            raise ValueError("dimension mismatch")
    else:
        parameters["facecolor"] = "None"
        pc = add_poly(mesh, **parameters)

    _, ax = plt.subplots()
    if show_vertices:
        ax.plot(mesh.vertices[:, 0], mesh.vertices[:, 1],
                marker=".", ls="", color="k")

    if values is not None:
        if mesh.nx == mesh.ny:
            xv = mesh.vertices[:, 0]
            yv = mesh.vertices[:, 1]
            idx = numpy.asarray(numpy.lexsort((xv[::-1], yv[::-1])))
            idx = numpy.flip(idx.reshape((mesh.nx+1, mesh.ny+1)), 1)
            plt.contourf(xv[idx], yv[idx], values[idx])
            parameters["facecolor"] = "None"
            pc = add_poly(mesh, **parameters)
            plt.colorbar()
        else:
            cell_values = numpy.mean(values[mesh.cells], 1)
            pc = add_poly(mesh, **parameters)
            pc.set_array(cell_values)
            plt.colorbar(pc)

    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect("equal")
    plt.show()


def reorder_counterclockwise(cells: numpy.ndarray):
    """
    Reorder cell topology so that V0->V1->V2->V3->V0
    forms a closed path and the resulting quadrilateral
    doest not have self-intersections.
    """
    perm = numpy.array([0, 1, 3, 2], dtype=numpy.int32)
    new_cell_order = cells[:, perm]
    return new_cell_order


def add_poly(mesh, **kwargs):
    """
    Add matplotlib PolyCollection, one polygon per cell.
    """
    ordered_cells = reorder_counterclockwise(mesh.cells)
    verts = mesh.vertices[ordered_cells]
    pc = matplotlib.collections.PolyCollection(verts, **kwargs)
    return pc


if __name__ == "__main__":
    simple_mesh = Mesh(30, 30)
    cell_values = numpy.arange(simple_mesh.num_cells)
    plot(simple_mesh, cell_values, False)
