import matplotlib.pyplot as plt
import matplotlib.collections
import numpy

from simple_fem import Mesh


def plot(mesh: Mesh, values=None, show_vertices=True):
    parameters = {"edgecolor": "k",
                  "cmap": "rainbow"}

    if values is not None:
        if values.size != mesh.num_cells:
            raise ValueError('dimension mismatch')
    else:
        parameters["facecolor"] = "None"

    fig, ax = plt.subplots()
    if show_vertices:
        ax.plot(mesh.vertices[:, 0], mesh.vertices[:, 1], marker="o", ls="", color='k')

    pc = add_poly(mesh, **parameters)
    ax.add_collection(pc)
    ax.autoscale()
    ax.set_aspect('equal')

    if values is not None:
        pc.set_array(values)
        fig.colorbar(pc, ax=ax)

    plt.show()

    # fi
    # return fig


def reorder_counterclockwise(cells: numpy.ndarray):
    """
    Reorder mesh to counter clock wise local
    """
    perm = numpy.array([0, 1, 3, 2], dtype=numpy.int32)
    new_cell_order = cells[:, perm]
    return new_cell_order


def add_poly(mesh, **kwargs):
    ordered_cells = reorder_counterclockwise(mesh.cells)
    verts = mesh.vertices[ordered_cells]
    pc = matplotlib.collections.PolyCollection(verts, **kwargs)
    return pc


if __name__ == '__main__':
    simple_mesh = Mesh(10, 10)
    values = numpy.arange(simple_mesh.num_cells)
    plot(simple_mesh, values)
