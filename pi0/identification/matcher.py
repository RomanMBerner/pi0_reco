import numpy as np
from scipy.spatial import distance

class Pi0Matcher():

    def find_matches(self, sh_starts, sh_directions, sh_energies, segment, method, tolerance=10., *ppn_track_points):
        '''
        Project 2 paired gammas back to their crossing point and return nearest 
        (within tolerance) track labeled hit as vertex candidate
        Inputs:
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - sh_energies (M x 3): array of shower energies
            - segment (N x 5): array of voxels and their semantic segmentation labels
        Returns:
            - Array of id pairs (one per pair of matched showers)
            - Array of production vertices
        '''
        if method == 'proximity':
            # Mask out points that are not track-like, return if there is no such point
            track_mask = np.where(segment[:,-1] < 2)[0]
            if not len(track_mask):
                return []
            track_data = segment[track_mask]

            # Get the best valid pairs (closest and compatible)
            matches, vertices, dists = self.find_best_pairs(np.array(sh_starts), np.array(sh_directions), tolerance)

            # Find the closest track-like point for each pair, adjust vertex position
            if len(vertices):
                vertices = self.find_closest_points(np.array(vertices), track_data[:,:3])

            return matches, vertices, dists
        
        elif method == 'ppn':
            
            matches, vertices = [], []
            if len(ppn_track_points[0])==0 or len(sh_starts)==0:
                return matches, vertices
            
            # For all showers: Find pair-wise POCAs and closest distance of a shower's direction to the POCAs
            POCAs = self.find_pair_wise_POCAs(np.array(sh_starts), np.array(sh_directions), tolerance)
            
            # For all POCAs: Find track-labeled PPN points close to a POCA
            ppns = []
            for point in ppn_track_points[0]:
                ppns.append([point.ppns[0], point.ppns[1], point.ppns[2]])
            vertex_candidates = self.select_POCAs_close_to_PPNs(POCAs, ppns, tolerance)
            if len(vertex_candidates)==0:
                return matches, vertices

            # Find showers with a direction approximatively (shower_start - vertex_candidate)
            spatial_tolerance = 5.  # if a sh_start is close to a vertex_candidate: take the candidate as sh_start
                                    # TODO: Read from config file?
                                    # TODO: Optimise this value (it's probably too small)
            angular_tolerance = 15. # the sh_direction must be (up to this tolerance) the same as (sh_start-vertex_candidate_pos)
                                    # TODO: Read from config file?
                                    # TODO: Optimise this value (it's probably too small)
            vertices = self.find_shower_vertices(np.array(sh_starts), np.array(sh_directions), np.array(sh_energies), vertex_candidates, spatial_tolerance, angular_tolerance)
            if len(vertices)==0:
                return matches, vertices

            # Match shower pairs to pi0 decays
            matches, vertices = self.find_photon_pairs(vertices)

            return matches, vertices

        else:
            raise ValueError('Shower matching method not recognized:', method)


    def find_best_pairs(self, sh_starts, sh_directions, tolerance):
        '''
        Finds the indices of the best pairs, compute the interaction vertex for each pair
        Inputs:
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - tolerance: defines the maximum distance between two lines of the shower directions at their closest point
        Returns:
            - Array of id pairs (one per pair of matched showers)
            - Array of production vertices
        '''
        # Get vertex (mean of POCAs) and the distance for each pair of showers, build a separation matrix
        npoints = len(sh_starts)
        sep_mat = np.full((npoints, npoints), float('inf'))
        ver_mat = [[0., 0., 0.] for _ in range(npoints*npoints)]
        for i in range(npoints):
            for j in range(i+1, npoints):
                vertex, dist = self.find_vertex([sh_starts[i], sh_starts[j]], [sh_directions[i], sh_directions[j]])
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


    def find_vertex(self, sh_starts, sh_directions):
        """
        Calculates the interaction vertex as the mean of the two POCAs
        and the minimum distance as the seperation between them.
        
        Inputs:
            - sh_starts (2 x 3): array of shower start points
            - sh_directions (2 x 3): array of shower directions
        Returns:
            - Interaction point
            - Shortest distance between the two lines
        """
        pocas = self.find_pocas(sh_starts, sh_directions)
        return np.mean(pocas,axis=0), np.linalg.norm(pocas[1]-pocas[0])


    @staticmethod
    def find_pocas(sh_starts, sh_directions):
        '''
        Calculates point of closest approach (POCA) between two vectors
        *in the backward direction*. If lines are parallel or point of 
        closest approach is in the "forward" direction returns the 
        separation between the two sh_starts.
        
        Inputs:
            - sh_starts (2 x 3): array of shower start points
            - sh_directions (2 x 3): array of shower directions (*normalized*)
        Returns:
            - Array of scalars for projection along each vector to reach POCA
            - Separation at point of closes approach (in backwards direction)
        '''
        # Get the angle between vectors
        d = sh_starts[0]-sh_starts[1]
        v0, v1 = sh_directions[0], sh_directions[1]
        v_dp = np.dot(v0, v1)

        # Check for parallel lines, POCA undefined if parrallel
        if abs(v_dp) == 1:
            raise ValueError('The two shower axes are parallel, cannot find POCA')

        # Minimize the distance
        s0 = (-np.dot(d,v0) + np.dot(d,v1)*v_dp)/(1-v_dp**2)
        s1 = ( np.dot(d,v1) - np.dot(d,v0)*v_dp)/(1-v_dp**2)

        # Check that we have propogated both vectors in the backward dir
        if s0 > 0 or s1 > 0:
            return [sh_starts[0], sh_starts[1]]

        # Minimum separation
        poca0 = sh_starts[0] + s0 * sh_directions[0]
        poca1 = sh_starts[1] + s1 * sh_directions[1]
        return [poca0, poca1]


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


    def find_pair_wise_POCAs(self, sh_starts, sh_directions, tolerance):
        '''
        Finds the indices of the best pairs, compute the interaction vertex for each pair
        Inputs:
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - tolerance: defines the maximum distance between two lines of the shower directions at their closest point
        Returns:
            - Array of POCAs (possible pi0 vertices)
        '''
        # Get POCAs between all shower pairs (and the distance between the two lines of the shower directions)
        POCAs = []
        npoints = len(sh_starts)
        for i in range(npoints):
            for j in range(i+1, npoints):
                POCA, dist = self.find_vertex([sh_starts[i], sh_starts[j]], [sh_directions[i], sh_directions[j]])
                if dist < tolerance:
                    POCAs.append(POCA)

        return np.array(POCAs)


    def select_POCAs_close_to_PPNs(self, POCAs, ppns, tolerance):
        '''
        Finds vertex_candidates = track-labeled PPN points which are close (distance < tolerance) to any POCA
        Inputs:
            - POCAs (N x 3): array of POCA points (3D coordinates)
            - ppns (M x 3): array of track-labeled points from PPN (3D coordinates)
            - tolerance: defines the maximum spatial distance between a PPN point and a POCA point
        Returns:
            - Array of vertex_candidates (possible pi0 vertices)
        '''        
        distances = distance.cdist(POCAs,ppns)

        # Get the closest track_labeled PPN point for each POCA.
        # If the distance between this point and the POCA is < tolerance, return it as vertex_candidate
        idxs = np.argmin(distance.cdist(POCAs,ppns),axis=1)

        used_indices = []
        vertex_candidates = []
        for i, idx in enumerate(idxs):
            if (distances[i][idx]<tolerance and idx not in used_indices):
                used_indices.append(idx)
                vertex_candidates.append(ppns[idx])

        return np.array(vertex_candidates)


    def find_shower_vertices(self, sh_starts, sh_directions, sh_energies, vertex_candidates, spatial_tolerance, angular_tolerance):
        '''
        Finds vertex_candidates = track-labeled PPN points which are close (distance < tolerance) to any POCA
        Inputs:
            - sh_starts (N x 3): array of shower start points
            - sh_directions (N x 3): array of shower directions
            - vertex_candidates (M x 3): array of vertex candidates
            - spatial_tolerance: if a sh_start is close to a vertex_candidate, take the vertex candidate as sh_vertex
            - angular_tolerance: defines the maximum difference between sh_direction and (sh_start-vertex_candidate) [in degrees]
        Returns:
            - Array of shower_vertices
        '''        
        vertices = []

        # For all showers which have sh_start close (up to spatial_tolerance) to a vertex_candidate:
        # Take the vertex_candidate as the shower's vertex
        dist = distance.cdist(sh_starts,vertex_candidates)
        idxs = np.argmin(distance.cdist(sh_starts,vertex_candidates),axis=1)

        used_sh_indices  = []
        used_sh_vertices = []
        for i, idx in enumerate(idxs):
            # If shower_start is very close (<spatial_tolerance) to vertex candidate, take vertex_candidate as pi0 decay vertex
            # TODO: Note: Showers with start position very close to the vertex likely are from electrons / positrons.
            # TODO: If it is likely to be e-/e+: Do not consider this shower and look for other showers instead.
            if (dist[i][idx]<spatial_tolerance):
                used_sh_indices.append(i)
                used_sh_vertices.append(vertex_candidates[idx])
                #print(' Take vertex ', vertex_candidates[idx], ' since it is close to shower ', i, ' with start point ', sh_starts[i])
                
            # If shower_start is close (<20 px) to vertex candidate and sh_energy is small (<40 MeV), take vertex_candidate as pi0 decay vertex
            # TODO: Optimise values for energy and distance
            #if (sh_energies[idx]<40. and dist[i][idx]<20):
            #    used_sh_indices.append(i)
            #    used_sh_vertices.append(vertex_candidates[idx])
            #    print(' Take vertex ', vertex_candidates[idx], ' since sh_energy ', sh_energies[idx], ' < 40 MeV and sh_start is close ( ', dist[i][idx], ' < 20 px) to shower ', i, ' with start point ', sh_starts[i])

        # For all other showers which do not have a vertex_candidate close enough:
        #Take as vertex vertex_candidate which (in combination with sh_start) is in best angular agreement with the sh_direction
        counter = 0
        for sh_index, sh_start in enumerate(sh_starts):
            if sh_index in used_sh_indices:
                vertices.append(used_sh_vertices[counter])
                counter += 1
            else:
                min_angle = float('inf')
                for vtx_index, vtx_pos in enumerate(vertex_candidates):
                    angle = np.arccos(np.dot((sh_start-vtx_pos)/np.linalg.norm(sh_start-vtx_pos),sh_directions[sh_index]))*360./(2.*np.pi)
                    if angle < min_angle:
                        selected_vtx_index = vtx_index
                        selected_vtx_pos = vtx_pos
                        min_angle = angle
                if min_angle >= angular_tolerance:
                    selected_vtx_index = None
                    selected_vtx_pos = np.array([None,None,None])
                vertices.append(selected_vtx_pos)

        return np.array(vertices)


    def find_photon_pairs(self, vertices):
        '''
        Finds matches (= pairs of showers originated from the same vertex)
        Inputs:
            - vertices (N x 3): array of vertex positions, N = number of showers in the event
        Returns:
            - matches: Array of shower_id pairs (one per pair of matched showers)
        '''
        # TODO: At the moment, only return pi0_vertices if exactly two showers point to the same track-labeled PPN point.
        #       This algorithm can be extended to the case where >2 showers originated from the same vertex.
        # TODO: Could be interesting to close the kinematics for the cases where >2 showers originated from the same vertex.
        matches = []

        # Loop over all vertices.
        # If a vertex is present exactly two times: return the two indices (corresponding to the shower_ids) in the vertex array
        for idx1, vtx1 in enumerate(vertices):
            counter = 0
            indices = []
            if not np.any(vtx1): # continue if vtx1 contains 'None'
                continue

            for idx2, vtx2 in enumerate(vertices):
                if not np.any(vtx2): # Continue if vtx2 contains 'None'
                    continue

                diff = vtx1- vtx2
                if np.all(diff==0):
                    #print(' vertices are equal: ', vtx1, ' = ', vtx2)
                    indices.append(idx2)

            if len(indices) == 2:
                if indices not in matches:
                    matches.append(indices)

        pi0_vertices = []
        for i, match in enumerate(matches):
            pi0_vertices.append(vertices[match[0]])

        return matches, pi0_vertices