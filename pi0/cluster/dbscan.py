import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Drop clusters with size < 30
# Return default vector if found no clusters


class DBSCANCluster:

    def __init__(self, eps=6.0, min_samples=5):
        '''
        Initialize the DBSCAN algorithm
        '''
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
        if len(shower_energy) < 1:
            return []
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
        clusts = []
        used_clusts = []
        for c in frags:
            if c == -1 or c in used_clusts:
                clusts.append([])
            else:
                ind = np.where(labels == c)[0]
                clusts.append(ind)
                used_clusts.append(c)
        return clusts, used_clusts


    def create_clusters(self, shower_energy, primaries=None):
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
        if primaries is not None:
            Y = cdist(coords, primaries[:, :3])
            min_dists, ind = np.min(Y, axis=1), np.argmin(Y, axis=0)
            frags = labels[ind]
            clusts, used_clusts = self.find_cluster_indices(coords, labels, frags)
            self._clusts = clusts
            remaining_labels = [l for l in np.unique(labels) if not l in used_clusts and l >=0]
            remaining_clusts = [np.where(labels == l)[0] for l in remaining_labels]
            remaining_energy = np.where(labels == -1)[0]
            return clusts, remaining_clusts, remaining_energy
        else:
            self._clusts = []
            remaining_clusts = [np.where(labels == l)[0] for l in np.unique(labels) if l != -1]
            remaining_energy = np.where(labels == -1)[0]
            return [], remaining_clusts, remaining_energy


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
