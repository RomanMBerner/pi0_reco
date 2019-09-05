import numpy as np
from scipy.spatial import distance

class Pi0Matcher():

    def find_matches(self, points, directions, segment, tolerance=10.):
        '''
        Project 2 paired gammas back to their crossing point and return nearest 
        (within tolerance) track labeled hit as vertex candidate

        Inputs:
            - points (M x 3): array of shower start points
            - directions (M x 3): array of shower directions
            - segment (N x 5): array of voxels and their semantic segmentation labels

        Returns:
            - Array of id pairs (one per pair of matched showers)
            - Array of production vertices
        '''
        # Mask out points that are not track-like, return if there is no such point
        track_mask = np.where(segment[:,-1] < 2)[0]
        if not len(track_mask):
            return []
        track_data = segment[track_mask]

        # Get the best valid pairs (closest and compatible)
        matches, vertices, dists = self.find_best_pairs(np.array(points), np.array(directions), segment, tolerance)

        # Find the closest track-like point for each pair, adjust vertex position
        if len(vertices):
            vertices = self.find_closest_points(np.array(vertices), track_data[:,:3])

        return matches, vertices, dists


    def find_closest_points(self, vertices, track_data):
        '''
        Find the closest track voxel to each candidate vertex, return their coordinates.

        Inputs:
            - vertices (M x 3): array of candidate vertex coordinates
            - track_data (N x 3): array of track-like voxel coordinates

        Returns:
            - Array of adjusted production vertices
        '''
        idxs = np.argmin(distance.cdist(vertices, track_data),axis=1)
        return track_data[idxs]


    def find_best_pairs(self, points, directions, segment, tolerance):
        '''
        Finds the indices of the best pairs, compute the interaction vertex for each pair

        Inputs:
            - points (M x 3): array of shower start points
            - directions (M x 3): array of shower directions
            - segment (N x 5): array of voxels and their semantic segmentation labels

        Returns:
            - Array of id pairs (one per pair of matched showers)
            - Array of production vertices
        '''
        # Get vertex (mean of POCAs) and the distance for each pair of showers,
        # Build a separation matrix.
        npoints = len(points)
        sep_mat = np.full((npoints, npoints), float('inf'))
        ver_mat = [[0., 0., 0.] for _ in range(npoints*npoints)]
        for i in range(npoints):
            for j in range(i+1, npoints):
                vertex, dist = self.find_vertex([points[i], points[j]], [directions[i], directions[j]])
                sep_mat[i,j] = dist
                ver_mat[i*npoints+j] = vertex

        # Iteratively find the closest match. Break if either:
        #  1. The mimimum distance is above the tolerance
        #  2. The number of remaining showers is below 2
        matches, vertices, dists = [], [], []
        while True:
            # Find the closest pair of points
            amin = np.argmin(sep_mat)
            idxs = np.unravel_index(amin, (npoints, npoints))
            sep = sep_mat[idxs]
            if sep > tolerance or sep == float('inf'):
                break

            # Record shower index pair, the vertex location and the distance between them
            matches.append(idxs)
            vertices.append(ver_mat[amin])
            dists.append(sep)

            # Max out the distance for the pairs involving those showers
            sep_mat[idxs,:] = sep_mat[:, idxs] = float('inf')

        return matches, vertices, dists


    def find_vertex(self, points, directions):
        """
        Calculates the interaction vertex as the mean of the two POCAs
        and the minimum distance as the seperation between them.
        
        Inputs:
            - points (2 x 3): array of shower start points
            - directions (2 x 3): array of shower directions

        Returns:
            - Interaction point
            - Shortest distance between the two lines
        """
        pocas = self.find_pocas(points, directions)
        return np.mean(pocas,axis=0), np.linalg.norm(pocas[1]-pocas[0])


    @staticmethod
    def find_pocas(points, directions):
        '''
        Calculates point of closest approach (POCA) between two vectors
        *in the backward direction*. If lines are parallel or point of 
        closest approach is in the "forward" direction returns the 
        separation between the two points.
        
        Inputs:
            - points (2 x 3): array of shower start points
            - directions (2 x 3): array of shower directions (*normalized*)

        Returns:
            - Array of scalars for projection along each vector to reach POCA
            - Separation at point of closes approach (in backwards direction)
        '''
        # Get the angle between vectors
        d = points[0]-points[1]
        v0, v1 = directions[0], directions[1]
        v_dp = np.dot(v0, v1)

        # Check for parallel lines, POCA undefined if parrallel
        if abs(v_dp) == 1:
            raise ValueError('The two shower axes are parallel, cannot find POCA')

        # Minimize the distance
        s0 = (-np.dot(d,v0) + np.dot(d,v1)*v_dp)/(1-v_dp**2)
        s1 = ( np.dot(d,v1) - np.dot(d,v0)*v_dp)/(1-v_dp**2)

        # Check that we have propogated both vectors in the backward dir
        if s0 > 0 or s1 > 0:
            return [points[0], points[1]]

        # Minimum separation
        poca0 = points[0] + s0 * directions[0]
        poca1 = points[1] + s1 * directions[1]
        return [poca0, poca1]
