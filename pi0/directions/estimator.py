import numpy as np
from mlreco.utils.gnn.cluster import form_clusters_new
from mlreco.utils.gnn.compton import filter_compton
from mlreco.visualization.voxels import scatter_label
from mlreco.utils.gnn.primary import assign_primaries_unique
from mlreco.utils import metrics

from sklearn.cluster import DBSCAN
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist

# Drop clusters with size < 30
# Return default vector if found no clusters


class FragmentEstimator:

    def __init__(self, eps=2.0, min_samples=5, min_energy=0.05):
        
        self._clusterer = DBSCAN(eps=eps, min_samples=min_samples)
        self._min_energy = min_energy
        self._coords = None
        self._labels = None
        self._clusts = None


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
        mask = shower_energy[:, -1] > self._min_energy
        coords = shower_energy[:, :3][mask]
        frag_labels = self._clusterer.fit_predict(coords)
        self._frag_labels = frag_labels
        return frag_labels, mask


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
            selected = coords[labels == c]
            ind = np.where(labels == c)
            centroid = np.mean(selected, axis=0)
            centroids.append(centroid)
            clusts.append(ind)
        centroids = np.asarray(centroids)
        self._centroids = centroids
        if len(clusts) == 0 and centroids.shape[0] == 0:
            print("No fragments were found for the supplied DBSCAN parameters")
        return clusts


    def assign_frags_to_primary(self, shower_energy, shower_segment_label, primaries, 
                                max_distance=float('inf')):
        """
        Inputs:
            - shower_energy (np.ndarray): energy depo array for SHOWERS ONLY
            - shower_segment_label (np.ndarray): segment_labels for SHOWERS ONLY
            - primaries (np.ndarray): primaries information from parse_em_primaries
            - max_distance (float): do not include voxels in fragments if distance from
            primary is larger than max_distance. 

        Returns:
            None (updates FragmentEstimator properties in-place)
        """
        labels, mask = self.make_shower_frags(shower_energy)
        coords = shower_energy[mask][:, :3]
        energies = shower_energy[mask][:, -1]
        #clusts = self.find_centroids_and_clusters(coords, labels)
        Y = cdist(coords, primaries[:, :3])
        min_dists, ind = np.min(Y, axis=1), np.argmin(Y, axis=0)
        frags = labels[ind]
        fragment_mask = np.isin(labels, frags)
        distance_mask = min_dists < max_distance
        mask2 = np.logical_and(fragment_mask, distance_mask)
        self._voxel_weights = energies[mask2]
        clusts = self.find_cluster_indices(coords[mask2], labels[mask2], frags)
        self._coords = coords[mask2]
        self._labels = labels[mask2]
        self._clusts = clusts
        self._primaries = primaries

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


class DirectionEstimator(FragmentEstimator):

    def __init__(self, eps=2, min_samples=5, min_energy=0.05):
        super().__init__(eps=eps, min_samples=min_samples, min_energy=min_energy)
        self._directions = None

    def get_directions(self, shower_energy, shower_segment_label, primaries, 
                       max_distance=float('inf'), mode='pca'):
        """
        Given data (see FragmentEstimator docstring), return estimated 
        unit direction vectors for each primary. 
        """
        self.assign_frags_to_primary(shower_energy, shower_segment_label, 
                                     primaries, max_distance=max_distance)
        directions = []
        if len(self.clusts) == 0:
            print("FragmentEstimator found no fragments, returning None")
            return None
        for i, p in enumerate(self.primaries[:, :3]):
            origin = p[:3]
            indices = self.clusts[i]
            if mode == 'pca':
                direction = self.pca_estimate(self.coords[indices])
                parity = self.compute_parity_flip(self.coords[indices], direction, origin=origin)
                direction *= parity
            elif mode == 'cent':
                weights = self.voxel_weights[indices]
                direction = self.centroid_estimate(self.coords[indices], p)
            else:
                raise ValueError('Invalid Direction Estimation Mode')
            directions.append(direction)
        directions = np.asarray(directions)
        directions = directions / np.linalg.norm(directions, axis=1).reshape(
            directions.shape[0], 1)
        return directions

    def centroid_estimate(self, coords, primary, weights=None):
        centroid = np.average(coords, axis=0, weights=weights)
        direction = centroid - primary
        return direction

    def pca_estimate(self, coords):
        fit = PCA(n_components=1).fit(coords)
        return fit.components_.squeeze(0)

    def compute_parity_flip(self, xyz_hit, vector, origin=(0,0,0)):
        '''
        Uses the dot product of the average xyz_hit vector and the specified 
        vector to determine if vector in same, opposite, or parallel to most hits
        Args:
            xyz_hit - an Nx3 array of hit locations
            vector - a length 3 comparison vector
            origin - an optional offset to remove from each xyz_hit vector
        Returns:
            -1 if vector and avg xyz_hit vector are in opposite directions and
            +1 if in the same direction. 0 if they are perpendicular
        '''
        xyz_d = xyz_hit - origin
        xyz_avg = np.mean(xyz_d, axis=0)
        dot = np.dot(xyz_avg, vector)
        if dot > 0:
            return +1.
        elif dot < 0:
            return -1.
        return 0.