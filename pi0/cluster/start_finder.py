import numpy as np
from sklearn.decomposition import PCA

class StartPointFinder:

    def __init__(self):
        pass

    def curv(self, voxels, voxel_id):
        """
        Finds the local curvature at the voxel 
        with voxel_id inside the set of voxels.
        The curvature is computed as the mean angle
        with respect to the mean direction.

        Inputs:
            - voxels (N x 3 array): voxel position array
            - voxel_id (int): index of the voxel

        Returns:
            - int: value of the local curvature
        """
        # Find the mean direction from that point
        refvox = voxels[voxel_id]
        axis = np.mean([v-refvox for v in voxels], axis=0)
        axis /= np.linalg.norm(axis)
        
        # Find the curvature
        return abs(np.mean([np.dot((voxels[i]-refvox)/np.linalg.norm(voxels[i]-refvox), axis) for i in range(len(voxels)) if i != voxel_id])) 

    def find_start_point(self, voxels):
        """
        Finds the start point of shower cluster by looking
        for the two points at the extreme of the principal
        axis and picking the one with largest curvature.

        Inputs:
            - voxels (N x 3 array): voxel position array
        Returns:
            - (N x 1 array): start point coordinates
        """
        # Find PCA
        pca = PCA()
        pca.fit(voxels)
        axis = pca.components_[0,:]
        
        # Compute coord values along that axis
        coords = [np.dot(v, axis) for v in voxels]
        ids = [np.argmin(coords), np.argmax(coords)]
        
        # Compute curvature of the
        curvs = [self.curv(voxels, ids[0]), self.curv(voxels, ids[1])]
        
        # Return ID of the point
        return voxels[ids[np.argmax(curvs)]]

    def find_start_points(self, voxels, clusts):
        """
        Finds the start point of shower cluster by looking
        for the two points at the extreme of the principal
        axis and picking the one with largest curvature.

        Inputs:
            - voxels (N x 3 array): voxel position array
            - clusts C x (N array): list of voxel ids that make
              up the clusters
        Returns:
            - (N x 3 array): start point coordinates
        """
        return np.vstack([self.find_start_point(voxels[c]) for c in clusts])


