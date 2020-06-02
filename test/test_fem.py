import pytest
import numpy

from simple_fem import Q1Element

def test_Q1Element():
    element = Q1Element()
    dof_coord = element.dof_coordinates
    for i in range(element.num_dofs):
        delta = numpy.zeros(element.num_dofs)
        delta[i] = 1
        assert numpy.allclose(element.basis(dof_coord[i, 0], dof_coord[i, 1]), delta)