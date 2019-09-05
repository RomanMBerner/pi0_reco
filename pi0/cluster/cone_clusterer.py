import numpy as np
from pi0.directions.estimator import *

class _Angle:
    """
    Angle and slope are dependent attributes, so we define a descriptor.
    This way slope and angle are properly linked, i.e., changing one of them
    will change the value of another consistently. 
    """
    def __get__(self, instance, owner):
        return np.arctan(instance.slope)
    def __set__(self, instance, value):
        instance.slope = np.tan(value)

class Cone:
    """
    Wrapper class for cone objects. 
    """
    angle = _Angle()

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
        """
        vecs = points - self.vertex
        pars = np.dot(vecs, self.direction)
        perps = np.linalg.norm(np.cross(vecs, self.direction), axis=1)
        cone_radii = pars * self.slope
        perps[pars < 0] *= -1
        mask = np.logical_and(perps < cone_radii, pars > 0)
        mask = np.logical_and(mask, pars < self.height)
        return mask
    
    def get_scores(self, points, norm=2):
        """
        Given N x 3 array of three-dimensional points, calculates the 
        embedding space distance score for clustering. 

        Inputs:
            - points (N x 3 array): 3d coordinates
            - norm: p for the p-norm to be used in embedding space metric calculation.
        
        Returns:
            - scores (N x 1 array): euclidean distance in (r,t) space,
            (see Cone::transform).
        NOTE: we assign -1 score to reject tachyons. 
        """
        embedding = self.transform(points)
        scores = np.linalg.norm(embedding, axis=1, ord=norm)
        return scores

    def transform(self, points):
        """
        Given spatial coordinate array, primary vertices, and cone,
        transform spatial coordinate to (r,t) space where:
            - r: dist(vertex, point) / cone_height.
            - t: angle(cone_axis, point) / cone_angle.
        
        Inputs:
            - coords (np.ndarray): N x 3 spatial coordinate array.
            - cone (Cone): Cone object

        Returns:
            - embedding (np.ndarray): N x 2 (r,t) coordinate array. 
        """
        r = np.linalg.norm(points - self.vertex, axis=1) / self.height
        pars = np.dot(points - self.vertex, self.direction)
        perps = np.linalg.norm(np.cross(points - self.vertex, self.direction), axis=1)
        cos = pars / (np.linalg.norm(points - self.vertex, axis=1) * np.linalg.norm(self.direction))
        angles = np.arccos(cos)
        # slopes = perps / pars
        # angles = np.arctan(slopes) / self.angle
        embedding = np.hstack((r.reshape(r.shape[0], 1), angles.reshape(angles.shape[0], 1)))
        return embedding

    def __repr__(self):
        name = "Cone [{}]".format(self.name)
        vertex = " - Vertex = ({0:.4f}, {1:.4f}, {2:.4f})".format(
                self.vertex[0], self.vertex[1], self.vertex[2])
        direction = " - Direction = ({0:.4f}, {1:.4f}, {2:.4f})".format(
                self.direction[0], self.direction[1], self.direction[2])
        height = " - Height = {0:.4f}".format(self.height)
        slope = " - Slope = {0:.4f}".format(self.slope)
        angle = " - Angle = {0:.4f} rads".format(self.angle)
        l = [name, vertex, direction, height, slope, angle]
        return '\n'.join(l)

    def __str__(self):
        return str(self.__repr__())

class ConeClusterer:
    """
    Clustering module using cone-clustering.
    """

    def __init__(self, dir_estimator, params):
        """
        Initializer for ConeCluster Class.

        Inputs:
            - dir_estimator (DirectionEstimator Object): DirectionEstimator module
            for predicting shower directions. 
            - params (dict): list of parameters specific to cone clustering. 
        """
        self._cones = []
        self.scale_height = params.get('scale_height', 14.107334041)
        self.scale_slope = params.get('scale_slope', 5.86322059)
        self.slope_percentile = params.get('slope_percentile', 53.0)
        self.dir_estimator = dir_estimator
        self.predict_mode = params.get('predict_mode', 'score')
        self.direction_mode = params.get('direction_mode', 'pca')

    def make_cone(self, coords, vertex, direction, name='None'):
        """
        Build Cone for each primary.

        Inputs:
            - coords (np.ndarray): N x 3 array of spatial coordinates
            - vertex (np.ndarray): (3,) array containing vertex coordinates
            - direction (np.ndarray): (3,) array containing the normalized axis
            direction vector from dir_estimator. 

        Returns:
            - cone (Cone Object): Cone object with the specified parameters. 
        """
        axis = np.mean(coords - vertex, axis=0) * self.scale_height
        height = np.linalg.norm(axis)
        pars = np.dot(coords - vertex, direction)
        perps = np.linalg.norm(np.cross(coords - vertex, direction), axis=1)
        slopes = perps / pars
        slope = np.percentile(slopes[slopes > 0], self.slope_percentile)
        cone = Cone(vertex[:3], direction, height, slope * self.scale_slope, name=name)
        return cone

    def fit_cones(self, shower_energy, primaries, mode='pca'):
        self._cones = []
        directions = self.dir_estimator.get_directions(shower_energy, primaries, mode=mode)
        axes = self.dir_estimator.get_directions(shower_energy, 
            primaries, mode='cent')
        for i, p in enumerate(primaries):
            vertex, axis, direction = p[:3], axes[i], directions[i]
            ind = self.dir_estimator.clusts[i]
            cone = self.make_cone(self.dir_estimator.coords[ind], vertex, directions[i])
            self._cones.append(cone)

    def fit_predict(self, shower_energy, primaries, mode='pca'):
        """
        For given N x 5 array of shower coordinate and energy depositions, run
        cone-clustering with the fitted cones.

        Inputs:
            - shower_energy (np.ndarray): N x 5 array, where first three are spatial
            coordinates and last column gives the energy depositions.
            - primaries (np.ndarray): N_p x 3 array of em primary vertices. 

        Returns:
            - pred (np.ndarray): (N, ) array of predicted cluster labels. 

        NOTE: 
        1) All points that do not belong inside any cones are given label -1.
        2) We calculate the score of a point belonging to a cone instance, using the
        euclidean distance calculated in transformed (r,t) space (see Cone::transform).
        3) We reject particles that violate causality, i.e., particles that must propagate
        backwards in time to lie within any cone. 

        """
        self.fit_cones(shower_energy, primaries, mode=mode)
        if self.predict_mode == 'contain':
            pred = -np.ones((shower_energy.shape[0], ))
            for i, cone in enumerate(self._cones):
                mask = cone.contains(shower_energy[:, :3])
                pred[mask] = i
            return pred
        elif self.predict_mode == 'score':
            scores = []
            for i, cone in enumerate(self._cones):
                s = cone.get_scores(shower_energy, norm=float('inf'))
                scores.append(s.reshape(s.shape[0], 1))
            scores = np.hstack(scores)
            self._scores = scores
            pred = np.argmin(scores, axis=1)
            return pred
        else:
            raise ValueError('Invalid Cone Prediction Mode: {}'.format(self.predict_mode))

    def adjust_cones(self, params):
        pass

    @property
    def cones(self):
        return self._cones

    @property
    def scores(self):
        return self._scores

