import numpy as np
from scipy.spatial import distance


class Pi0Matcher():

    def __init__(self, **cfg):
        '''
        Extracts the Pi0 matching parameters from the module configuration
        '''
        self._min_energy     = cfg.get('min_energy', 10)                  # Minimum shower energy in MeV
        self._select_photons = cfg.get('select_photons', False)           # Only use showers classified as photon-induced
        self._min_photon_lr  = cfg.get('min_photon_lr', 0.5)              # Minimum photon score to be classified as photon
        self._apply_fiducial = cfg.get('apply_fiducial', False)           # Only uses showers that stay inside the fiducial volume

        self._match_to_track = cfg.get('match_to_track', False)           # Pick the track point that minimizes angular disagreement
        self._match_to_ppn   = cfg.get('match_to_ppn', True)              # Pick the PPN track point that minimizes angular disagreement
        self._max_angle      = np.radians(cfg.get('max_angle', 10))       # Maximum angular disagreement in degrees

        self._angular_spread = np.radians(cfg.get('angular_spread', 2.1)) # Uncertainty on shower direction at 1 GeV in degrees
        self._mean_free_path = cfg.get('mean_free_path', 18.129)          # Photon asymptotic mean free path in cm
        self._voxel_size     = cfg.get('voxel_size', 0.3)                 # Image voxel size in cm

    def find_matches(self, showers, track_points=None, ppn_track_points=None):
        '''
        Matches gammas by finding their most likely crossing point. The likelihood
        of a crossing is curently purely angle-based and assumes identical angular
        uncertainty on both showers.

        Inputs:
            - showers (M x 1): array of shower objects (as defined in chain.py)
            - track_points (N x 3): array of track voxels
            - ppn_track_points (K x 3): array of track-labeled PPN points (K points, 3 coordinates each)
        Returns:
            - matches: Array of id pairs (one per pair of matched showers)
            - vertices: Array of the corresponding pi0 decay vertices
        '''
        # Extract reconstructed shower information, reject showers below the size threshold.
        # If requested, reject showers that are labeled as electron by the particle identifier
        sh_energies = np.array([s.energy for s in showers])
        mask = sh_energies > self._min_energy
        if self._select_photons:
            sh_photon_ls = np.array([s.L_p for s in showers])
            mask &= sh_photon_ls > self._min_photon_lr
        if self._apply_fiducial:
            sh_fiducials = np.array([s.fiducial for s in showers])
            mask &= sh_fiducials
        mask = np.where(mask)[0]
        ns   = len(mask)
        if ns < 2: return [], [], []

        sh_energies = sh_energies[mask]
        sh_starts = np.vstack([showers[i].start for i in mask])
        sh_directions = np.vstack([showers[i].direction for i in mask])

        # Find the most likely vertices and angles for each pair of showers
        sep_mat = np.full((ns, ns), float('inf'))
        ver_mat = [[0., 0., 0.] for _ in range(ns*ns)]
        for i in range(ns):
            for j in range(i+1, ns):
                matched = False
                starts, dirs = [sh_starts[i], sh_starts[j]], [sh_directions[i], sh_directions[j]]
                if self._match_to_ppn:
                    assert ppn_track_points is not None, 'Need PPN track points to match to them'
                    if len(ppn_track_points):
                        angles = [self.angular_displacement(starts, dirs, v) for v in ppn_track_points]
                        vertex = ppn_track_points[np.argmin(angles)]
                        angle  = np.min(angles)
                        matched = True
                if self._match_to_track and not matched:
                    assert track_points is not None, 'Need track points to match to track'
                    if len(track_points):
                        angles = [self.angular_displacement(starts, dirs, v) for v in track_points]
                        vertex = track_points[np.argmin(angles)]
                        angle  = np.min(angles)
                        matched = True
                if not matched:
                    vertex = self.find_crossing([sh_starts[i], sh_starts[j]], [sh_directions[i], sh_directions[j]])
                    angle  = self.angular_displacement([sh_starts[i], sh_starts[j]], [sh_directions[i], sh_directions[j]], vertex)

                ver_mat[i*ns+j] = vertex
                sep_mat[i,j] = angle

        # Match the showers together if they their angular agreement is good enough
        matches, vertices, angles = self.find_best_pairs(ver_mat, sep_mat, self._max_angle)

        # Convert matches back to original shower id
        if len(matches):
            matches = mask[np.vstack(matches)]

        return matches, vertices, angles

    def find_best_pairs(self, ver_mat, sep_mat, tolerance):
        '''
        Finds the indices of the best, compatible pairs of showers.

        Inputs:
            - ver_mat (M x M x 3): matrix of shower vertices, one per possible pair of showers
            - sep_mat (M x M): matrix of angular disagreement between the possible pairs
            - tolerance: defines the max. allowed angular disagreement
        Returns:
            - Array of id pairs (one per pair of matched showers)
            - Array of production vertices
        '''

        # Iteratively find the closest match. Break if either:
        #  1. The mimimum distance is above the tolerance
        #  2. The number of remaining showers is below 2
        matches, vertices, angles = [], [], []
        while True:
            # Find the closest pair of points
            amin = np.argmin(sep_mat)
            idxs = np.unravel_index(amin, sep_mat.shape)
            sep = sep_mat[idxs]
            if sep > tolerance or sep == float('inf'):
                break

            # Record shower index pair, the vertex location and the distance between them
            matches.append(idxs)
            vertices.append(ver_mat[amin])
            angles.append(sep)

            # Max out the distance for the pairs involving those showers
            sep_mat[idxs,:] = sep_mat[:, idxs] = float('inf')

        return matches, vertices, angles

    def find_crossing(self, sh_starts, sh_directions):
        """
        Calculates the interaction vertex as the point along the line joining
        the CPAs that minimizes the angular disagreement.

        Inputs:
            - sh_starts (2 x 3): array of shower start points
            - sh_directions (2 x 3): array of shower directions
        Returns:
            - Interaction point
            - Smallest angular disagreement between the two lines
        """
        # Find the closest points of approach
        CPAs = self.find_CPAs(sh_starts, sh_directions)

        # Pick the point between the CPAs that gives equal angular displacement for both showers
        d0, d1 = np.linalg.norm(sh_starts[0]-CPAs[0]), np.linalg.norm(sh_starts[1]-CPAs[1])
        if d0 == 0 or d1 == 0:
            vertex = np.mean(CPAs, axis=0)
        else:
            vertex = np.average(CPAs, axis=0, weights=[d1, d0]) # Larger distance, smaller weight
        return vertex

    def angular_displacement(self, sh_starts, sh_directions, vertex):
        """
        Given two showers and a vertex, finds the mean angular displacement between
        the lines joining the vertex to the shower starts and the shower directions.

        Inputs:
            - sh_starts (2 x 3): array of shower start points
            - sh_directions (2 x 3): array of shower directions
        Returns:
            - Interaction point
            - Smallest angular disagreement between the two lines
        """
        d0, d1 = np.linalg.norm(sh_starts[0]-vertex), np.linalg.norm(sh_starts[1]-vertex)
        angle  = 0.
        if d0 > 0:
            v0 = (sh_starts[0]-vertex)/np.linalg.norm(sh_starts[0]-vertex)
            angle += np.arccos(np.dot(sh_directions[0], v0))/2
        if d1 > 0:
            v1 = (sh_starts[1]-vertex)/np.linalg.norm(sh_starts[1]-vertex)
            angle += np.arccos(np.dot(sh_directions[1], v1))/2

        return angle

    @staticmethod
    def find_CPAs(sh_starts, sh_directions):
        '''
        Calculates closest point of approach (CPA) between two vectors
        *in the backward direction*:
        - If the CPA is in the forward direction for one of the two vectors,

        Inputs:
            - sh_starts (2 x 3): array of shower start points
            - sh_directions (2 x 3): array of shower directions (*normalized*)
        Returns:
            - Array of scalars for projection along each vector to reach CPA
            - Separation at closest point of approach (in backwards direction)
        '''
        # Get the angle between vectors
        d = sh_starts[0]-sh_starts[1]
        v0, v1 = sh_directions[0], sh_directions[1]
        v_dp = np.dot(v0, v1)

        # If lines are parallel, pick the points equidistant from the two starts.
        # Else, find the constants that minimize the line-to-line distance
        if abs(v_dp) == 1:
            s0, s1 = -np.dot(d,v0)/2, np.dot(d,v1)/2
        else:
            s0 = (-np.dot(d,v0) + np.dot(d,v1)*v_dp)/(1-v_dp**2)
            s1 = ( np.dot(d,v1) - np.dot(d,v0)*v_dp)/(1-v_dp**2)

        # If both constants are in the forward direction, pick start points
        if s0 > 0 and s1 > 0:
            return [sh_starts[0], sh_starts[1]]

        # If only one of the constants is, pick the corresponding shower start
        if s0 > 0:
            return [sh_starts[0], sh_starts[0]]
        if s1 > 0:
            return [sh_starts[1], sh_starts[1]]

        # Minimum separation
        CPA0 = sh_starts[0] + s0 * sh_directions[0]
        CPA1 = sh_starts[1] + s1 * sh_directions[1]
        return [CPA0, CPA1]
