from .mesh import Mesh


class DofMap:
    def __init__(self, mesh: Mesh):
        self.mesh = mesh
