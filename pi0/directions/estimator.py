import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Drop clusters with size < 30
# Return default vector if found no clusters


class FragmentEstimator:

    def __init__(self, eps=6.0, min_samples=5):
        
        self._clusterer = DBSCAN(eps=eps, min_samples=min_samples)


    def make_shower_frags(self, shower_energy):
        """
        Cluster showers for initial identification of shower stem using DBSCAN.

        NOTE: Batch size should be one. 

        Inputs: 
            - shower_energy (N x 5 array): energy deposition array where 
            the first three coordinates give the spatial indices and the last
            column gives the energy depositions. 

        Returns:
            - frag_labels: labels assigned by DBSCAN clustering.
            - mask: energy thresholding mask. 
        """
        coords = shower_energy[:, :3]
        frag_labels = self._clusterer.fit_predict(coords)
        return frag_labels


    def find_cluster_indices(self, coords, labels, frags):
        """
        Find the index labels for each fragment assigned to a primary. 

        Inputs:
            - coords (N x 3): spatial coordinate array
            - labels (N x 1): fragment labels
            - frags: the fragment label associated to the ith em primary. 

        Returns:
            - clusts (list of arrays): list of variable length arrays where 
            the ith array contains the index location of the fragment assigned to
            the ith primary. 
        """
        centroids = []
        clusts = []
        for c in frags:
            if c == -1: continue
            ind = np.where(labels == c)[0]
            clusts.append(ind)
        return clusts


    def assign_frags_to_primary(self, shower_energy, primaries):
        """
        Inputs:
            - shower_energy (np.ndarray): energy depo array for SHOWERS ONLY
            - primaries (np.ndarray): primaries information from parse_em_primaries
            - max_distance (float): do not include voxels in fragments if distance from
            primary is larger than max_distance. 

        Returns:
            None (updates FragmentEstimator properties in-place)
        """
        labels = self.make_shower_frags(shower_energy)
        coords = shower_energy[:, :3]
        self._coords = coords
        Y = cdist(coords, primaries[:, :3])
        min_dists, ind = np.min(Y, axis=1), np.argmin(Y, axis=0)
        frags = labels[ind]
        clusts = self.find_cluster_indices(coords, labels, frags)
        self._clusts = clusts
        return clusts

    def set_labels(self):
        labels = -np.ones((self.coords.shape[0], ))
        for i, ind in  enumerate(self.clusts):
            labels[ind] = i
        self._labels = labels

    @property
    def coords(self):
        return self._coords

    @property
    def labels(self):
        return self._labels

    @property
    def clusts(self):
        return self._clusts

    @property
    def primaries(self):
        return self._primaries

    @property
    def voxel_weights(self):
        return self._voxel_weights


class DirectionEstimator():

    def __init__(self):
        self._directions = None

    def get_directions(self, shower_energy, primaries, fragments,
                       max_distance=float('inf'), mode='pca', normalize=True, weighted=False):
        """
        Given data (see FragmentEstimator docstring), return estimated 
        unit direction vectors for each primary. 

        Inputs:
            - shower_energy (np.ndarray): N x 5 shower energy depositions
            - primaries (np.ndarray): N x 3 em primary coordinates
            - fragments (list of np.ndarray): fragments[i] is a array of primary fragment
            indices for the i-th primary. 
            - max_distance: maximum distance cut for primary fragment
            - mode: method for estimating direction, choose 'pca' or 'cent'
            - normalize: True returns unit direction vectors. 
            - weighted: If True, computes the weighted centroid. 
        """
        directions = []
        for i, p in enumerate(primaries[:, :3]):
            origin = p[:3]
            indices = fragments[i]
            coords = shower_energy[indices, :3]
            if mode == 'pca':
                direction = self.pca_estimate(coords)
                parity = self.compute_parity_flip(coords, direction, origin)
                direction *= parity
            elif mode == 'cent':
                weights = None
                if weighted:
                    weights = shower_energy[indices, -1]
                direction = self.centroid_estimate(coords, p, weights)
            else:
                raise ValueError('Invalid Direction Estimation Mode')
            directions.append(direction)
        directions = np.asarray(directions)
        if normalize:
            directions = directions / np.linalg.norm(directions, axis=1).reshape(
                directions.shape[0], 1)
        self._directions = directions
        return directions

    def centroid_estimate(self, coords, primary, weights=None):
        centroid = np.average(coords, axis=0, weights=weights)
        direction = centroid - primary
        return direction

    def pca_estimate(self, coords):
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
