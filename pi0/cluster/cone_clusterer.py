import numpy as np


class Cone:
    """
    Wrapper class for cone objects. 
    """
    def __init__(self, vertex, direction, height, slope, name=None):
        """
        Initialize Cone, using vertex, direction (unit vector), height and slope.
        
        NOTE: The direction must point AWAY from the vertex. 

        Inputs:
            - vertex: (3, ) array
            - direction: (3, ) array
            - height: float number
            - slope: float number 
        """
        self.vertex = vertex
        self.direction = direction
        self.height = height
        self.slope = slope
        if name is not None:
            self.name = name
        else:
            self.name = ''
    
    def contains(self, points):
        """
        Given N x 3 array of three-dimensional points, determines if a each point
        lies within the cone.

        Inputs:
            - points (N x 3 array): 3d coordinates
        
        Returns:
            - mask (N x 1 array): Truth if given point is inside the cone. 
            - scores (N x 1 array): 
        """
        vecs = points - self.vertex
        par_axis = np.dot(vecs, self.direction)
        perp_axis = np.cross(vecs, self.direction)
        cone_radii = par_axis * self.slope
        scores = np.linalg.norm(perp_axis, axis=1)
        scores[par_axis < 0] *= -1
        mask = np.logical_and(scores < cone_radii, par_axis > 0)
        return mask, scores

    def __repr__(self):
        name = "Cone [{}]".format(self.name)
        vertex = " - Vertex = ({0:.4f}, {1:.4f}, {2:.4f})".format(
                self.vertex[0], self.vertex[1], self.vertex[2])
        direction = " - Direction = ({0:.4f}, {1:.4f}, {2:.4f})".format(
                self.direction[0], self.direction[1], self.direction[2])
        height = " - Height = {0:.4f}".format(self.height)
        slope = " - Slope = {0:.4f}".format(self.slope)
        l = [name, vertex, direction, height, slope]
        return '\n'.join(l)

class ConeClusterer:
    """
    Clustering module using cone-clustering.
    """

    def __init__(self, params=[]):
        self._cones = []

    def make_cones(self, coords, primaries, directions):
        pass

    def cluster_points(self, coords):
        pass

    def adjust_cones(self, params):
        pass

