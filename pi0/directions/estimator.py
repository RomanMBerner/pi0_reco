import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

class DirectionEstimator():

    def __init__(self, **cfg):
        """
        Extracts the direction esimatator parameters from the module configuration
        """
        self._directions   = None

        self._mode         = cfg.get('mode', 'cent')    # Method for estimating direction
        self._min_distance = cfg.get('min_distance', 5) # Minimum neighborhood radius to consider to estimate direction
        self._max_distance = cfg.get('max_distance', 5) # Neighborhood radius used to estimate direction if not optimize
        self._optimize     = cfg.get('optimize', False) # Optimize the neighborhood radius to minimize transverse spread
        self._normalize    = cfg.get('normalize', True) # Normalize direction vector length to 1
        self._weighted     = cfg.get('weighted', False) # Give move weight to voxels with a higher value

        modes = ['cent', 'pca']
        assert self._mode in modes,\
            f'Direction estimation mode {self._mode} not recognized, must be one of {modes}'

    def get_directions(self, primaries, fragments):
        """
        Return estimated unit direction vectors for each primary.

        Inputs:
            - primaries (np.ndarray): N x 3 em primary coordinates
            - fragments (list of np.ndarray): fragments[i] is (N,3+M+1) shower energy depositions (x,y,z,...,E)
            indices for the i-th primary.
        """
        # Loop over start points
        directions = []
        for i, p in enumerate(primaries[:, :3]):

            origin = p[:3]
            coords = fragments[i][:,:3]

            # If a maximum distance from the fragment origin is specified, down select points
            if not self._optimize and self._max_distance > 0:
                minid = np.argmin(cdist(coords, [origin]).flatten())
                dists = cdist(coords, [coords[minid]]).flatten()
                dist_mask = np.where(dists < self._max_distance)[0]
                coords = coords[dist_mask]

            # If optimization is required, find neighborhood that minimizes relative transverse spread
            elif self._optimize:
                # Order the cluster points by increasing distance to the start point
                minid = np.argmin(cdist(coords, [origin]).flatten())
                dist_mat = cdist(coords, [coords[minid]]).flatten()
                order = np.argsort(dist_mat)
                coords = coords[order]
                dist_mat = dist_mat[order]

                # If a minimum distance is specified, find which is the first point to consider
                min_id = 2
                if self._min_distance > 0:
                    inside_mask = np.where(dist_mat < self._min_distance)[0]
                    if len(inside_mask):
                        min_id = inside_mask[-1]

                # Find the PCA relative secondary spread for each point
                labels = np.zeros(len(coords))
                meank = np.mean(coords[:min_id+1], axis=0)
                covk = (coords[:min_id+1]-meank).T.dot(coords[:min_id+1]-meank)/(min_id+1)
                for i in range(min_id, len(coords)):
                    # Get the eigenvalues and eigenvectors, identify point of minimum secondary spread
                    w, _ = np.linalg.eigh(covk)
                    labels[i] = np.sqrt(w[2]/(w[0]+w[1])) if (w[0]+w[1]) else 0.
                    if dist_mat[i] == dist_mat[i-1]:
                        labels[i-1] = 0.

                    # Increment mean and matrix
                    if i != len(coords)-1:
                        meank = ((i+1)*meank+coords[i+1])/(i+2)
                        covk = (i+1)*covk/(i+2) + (coords[i+1]-meank).reshape(-1,1)*(coords[i+1]-meank)/(i+1)

                # Subselect coords that are most track-like
                max_id = np.argmax(labels)
                coords = coords[:max_id+1]

            # If PCA is required, return the principal component
            if self._mode == 'pca':
                direction = self.pca_estimate(coords)
                parity = self.compute_parity_flip(coords, direction, origin)
                direction *= parity

            # If centroid is required, return the mean distance wrt to the start point
            elif self._mode == 'cent':
                weights = None
                if self._weighted:
                    weights = fragments[:, -1]
                direction = self.centroid_estimate(coords, origin, weights)

            directions.append(direction)

        # Normalize direction to norm 1 if requested
        directions = np.asarray(directions)
        if self._normalize:
            directions = directions / np.linalg.norm(directions, axis=1).reshape(
                directions.shape[0], 1)
        self._directions = directions

        return directions

    def centroid_estimate(self, coords, primary, weights=None):
        '''
        Computes the voxel neighborhood centroid and its displacement from the start point
        '''
        centroid = np.average(coords, axis=0, weights=weights)
        direction = centroid - primary
        return direction

    def pca_estimate(self, coords):
        '''
        Computes the principal axis of the neighborhood around the start point
        '''
        fit = PCA(n_components=1).fit(coords)
        return fit.components_.squeeze(0)

    def compute_parity_flip(self, coords, direction, origin):
        '''
        Uses the dot product of the average vector and the specified
        vector to determine if vector in same, opposite, or parallel to most hits

        Inputs:
            - coords (N x 3): spatial coordinate array of the points in the primary cluster
            - direction (1 x 3): principal axis as set by the PCA
            - origin (1 x 3): starting point of the cluster

        Returns:
            - parity sign
        '''
        centered = coords - origin
        mean = np.mean(centered, axis=0)
        dot = np.dot(mean, direction)
        return np.sign(dot)

    @property
    def directions(self):
        return self._directions
