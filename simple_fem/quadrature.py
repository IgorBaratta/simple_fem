from scipy.special.orthogonal import ps_roots
import numpy


class Quadrature:
    """
    Computes the sample points and weights for Gauss-Legendre quadrature on a unit square.
    """

    def __init__(self, order):
        """
        Parameters:
        ----------
        order:
            Quadrature Order

        """
        self._order = order

        # The integration rules for the unit square can be constructed
        # by taking tensor products of the standard one-dimensional
        # Gauss-Legendre quadrature
        x, w = ps_roots(self._order)
        self._size = x.size * x.size
        self._points = (
            numpy.array(numpy.meshgrid(x, x, indexing="ij"))
            .transpose()
            .reshape(self._size, 2)
        )
        self._weights = numpy.outer(w, w).reshape((self._size, 1))

    @property
    def points(self):
        """
        Return Gauss-Legendre quadrature points.
        """
        return self._points

    @property
    def weights(self):
        """
        Return Gauss-Legendre quadrature points.
        """
        return self._weights

    @property
    def size(self):
        """
        Return the number of quadrature points.
        """
        return self._size
