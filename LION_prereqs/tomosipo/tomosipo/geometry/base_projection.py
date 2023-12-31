from .transform import Transform
import tomosipo as ts


def is_projection(g):
    """Determine if a geometry is a projection geometry

    A geometry object can be a volume geometry or a projection
    geometry.

    :param g: a geometry object
    :returns: `True` if `g` is a projection geometry
    :rtype: bool

    """
    return isinstance(g, ProjectionGeometry)


def is_cone(g):
    return is_projection(g) and g.is_cone


def is_parallel(g):
    return is_projection(g) and g.is_parallel


class ProjectionGeometry(object):
    """A general base class for projection geometries"""

    def __init__(self, shape=1):
        """Create a projection geometry

        :param shape:
        :returns:
        :rtype:

        """
        height, width = ts.types.to_shape2d(shape)

        self._shape = (height, width)

    ###########################################################################
    #                               __dunders__                               #
    ###########################################################################
    def __repr__(self):
        raise NotImplementedError()

    def __eq__(self, other):
        raise NotImplementedError()

    def __len__(self):
        return self.num_steps

    ###########################################################################
    #                               to_* methods                              #
    ###########################################################################
    def to_astra(self):
        """Convert geometry to astra geometry

        :returns: An astra representation of the current geometry.
        :rtype: dict

        """

        raise NotImplementedError()

    def to_vec(self):
        """Return a vector geometry of the current geometry

        :returns:
        :rtype: ProjectionGeometry

        """
        raise NotImplementedError()

    ###########################################################################
    #                                Properties                               #
    ###########################################################################
    @property
    def is_cone(self):
        """Is this geometry a cone-beam geometry?

        :returns:
        :rtype:

        """
        return self._is_cone

    @property
    def is_parallel(self):
        """Is this geometry a parallel-beam geometry?

        :returns:
        :rtype:

        """
        return self._is_parallel

    @property
    def is_vec(self):
        """Is this a vector geometry?

        A geometry can either be a vector geometry, like
        ``ConeVectorGeometry``, or ``ParallelVectorGeometry``, or it
        can be a parametrized geometry like ``ConeGeometry`` or
        ``ParallelGeometry``.

        :returns:
        :rtype:

        """
        return self._is_vector

    @property
    def det_shape(self):
        """The shape of the detector.

        :returns: `(int, int)`
            A tuple describing the height and width of the detector in
            pixels.
        :rtype: np.array

        """
        return self._shape

    @property
    def num_angles(self):
        """The number of angles in the projection geometry

        :returns:
            The number of angles in the projection geometry.
        :rtype: integer

        """
        raise NotImplementedError()

    @property
    def num_steps(self):
        """The number of orientations in the geometry

        This property equals `num_angles` and is provided so
        projection geometries and volume geometries share the same
        interface.

        :returns:
        :rtype:

        """
        return self.num_angles

    @property
    def angles(self):
        """The angles from which projections are acquired.

        This property is specific to non-vector geometries.

        :returns:
            A numpy array of shape with length `num_angles` containing
            the angles in radians.
        :rtype: `np.array`

        """
        raise NotImplementedError()

    @property
    def src_pos(self):
        """The source positions of the geometry.

        Not supported on parallel geometries.

        :returns: `np.array`
            A `(num_angles, 3)`-shaped numpy array containing the
            (Z,Y,X)-coordinates of the source positions.
        :rtype:

        """

        raise NotImplementedError()

    @property
    def det_pos(self):
        """The detector positions of the geometry.

        :returns: `np.array`
            A `(num_angles, 3)`-shaped numpy array containing the
            (Z,Y,X)-coordinates of the detector positions.
        :rtype:

        """
        raise NotImplementedError()

    @property
    def det_v(self):
        """The detector v-vectors of the geometry.

        The 'v' vector is usually the "upward" pointing vector
        describing the distance from the (0, 0) to (1, 0) pixel.

        :returns: `np.array`
            A `(num_angles, 3)`-shaped numpy array containing the
            (Z,Y,X)-coordinates of the v vectors.
        :rtype:

        """
        raise NotImplementedError()

    @property
    def det_u(self):
        """The detector u-vectors of the geometry.

        The 'u' vector is usually the "sideways" pointing vector
        describing the distance from the (0, 0) to (0, 1) pixel.

        :returns: `np.array`
            A `(num_angles, 3)`-shaped numpy array containing the
            (Z,Y,X)-coordinates of the u vectors.
        :rtype:

        """
        raise NotImplementedError()

    @property
    def det_normal(self):
        """The detector normal vectors of the geometry.

        In tomosipo, the normal vector usually points "into" the
        detector and away from the source/object. It is orthogonal to
        the `v` and `u` vectors.

        :returns: `np.array`
            A `(num_angles, 3)`-shaped numpy array containing the
            (Z,Y,X)-coordinates of the normal vectors.
        :rtype:

        """
        raise NotImplementedError()

    @property
    def ray_dir(self):
        """The ray direction of the geometry.

        This property is not supported on cone geometries.

        :returns: `np.array`
            A `(num_angles, 3)`-shaped numpy array containing the
            (Z,Y,X)-coordinates of the ray direction vectors.
        :rtype:

        """
        raise NotImplementedError()

    @property
    def det_size(self):
        """The size of the detector.

        This property is dangerous for vector geometries, because for
        vector geometries it is not guaranteed that the detector size
        is constant through time. If the detector size is not
        constant, this property raises a ValueError. Otherwise, it
        returns the detector size.

        :returns:
            The absolute size of the detector in the `v` and the `u`
            direction.
        :rtype: (scalar, scalar)

        """
        raise NotImplementedError()

    @property
    def det_sizes(self):
        """The size of each detector.

        :returns: np.array
            Array with shape (num_angles, 2) in v and u direction
            (height x width)
        :rtype: np.array

        """
        raise NotImplementedError()

    @property
    def corners(self):
        """Returns a vector with the corners of each detector

        :returns: np.array
            Array with shape (num_angles, 4, 3), describing the 4
            corners of each detector in (Z, Y, X)-coordinates.
        :rtype: np.array
        """
        raise NotImplementedError()

    @property
    def lower_left_corner(self):
        """Returns a vector with the positions of the lower-left corner the detector

        :returns: np.array
            Array with shape (num_angles, 3), describing the position
            of the lower-left corner of each detector in (Z, Y,
            X)-coordinates.
        :rtype: np.array

        """
        raise NotImplementedError()

    ###########################################################################
    #                          Transformation methods                         #
    ###########################################################################
    def rescale_det(self, scale):
        """Rescale detector pixels

        Rescales detector pixels without changing the size of the detector.

        :param scale: `int` or `(int, int)`
            Indicates how many times to enlarge a detector pixel. Per
            convention, the first coordinate scales the pixels in the
            `v` coordinate, and the second coordinate scales the
            pixels in the `u` coordinate.
        :returns: a rescaled geometry
        :rtype: `ProjectionGeometry`

        """
        raise NotImplementedError()

    def reshape(self, new_shape):
        """Reshape detector pixels without changing detector size

        :param new_shape: int or (int, int)
            The new shape of the detector in pixels in `v` (height)
            and `u` (width) direction.
        :returns: `self`
        :rtype: ProjectionGeometry

        """
        raise NotImplementedError()

    def project_point(self, point):
        """Projects point onto detectors

        This function projects onto the virtual detectors described by
        the projection geometry.

        For each source-detector pair, this function returns

            (detector_intersection_y, detector_intersection_x)

        where the distance is from the detector origin (center) and
        the units are in detector pixels.

        This function returns `np.nan` if the source-point ray is
        parallel to the detector plane.

        This function projects onto the detector even if the ray is
        moving away from the detector instead of towards it, or if the
        detector is between the source and point instead of behind the
        point.

        :param point: A three-dimensional vector (preferably np.array)
        :returns: np.array([[detector_intersection_y, detector_intersection_x],
                            .....])
        :rtype: np.array (num_angles * 2)
        """

        raise NotImplementedError()

    def __rmul__(self, other):
        """Applies a projective matrix transformation to geometry

        :param other: `np.array`
            A transformation matrix
        :returns: A transformed geometry
        :rtype: `ProjectionGeometry`

        """

        if isinstance(other, Transform):
            raise NotImplementedError()

        return NotImplemented
