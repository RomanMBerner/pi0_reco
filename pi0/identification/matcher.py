import numpy as np
from scipy.spatial import distance


class Pi0Matcher():

    def find_matches(self, reco_showers, segment, method, verbose, *ppn_track_points):
        '''
        Project 2 paired gammas back to their crossing point and return nearest 
        (within tolerance) track labeled hit as vertex candidate
        Inputs:
            - reco_showers (M x 1): array of reco_shower objects (as defined in chain.py)
            #- sh_starts (M x 3): array of shower start points
            #- sh_directions (M x 3): array of shower directions
            #- sh_energies (M x 3): array of shower energies
            - segment (N x 5): array of voxels and their semantic segmentation labels
            - method: proximity or ppn
            - verbose: if more printouts should be made
            - ppn_track_points (K x 5): array of track-labeled PPN points (K points, 3 coordinates + 1 score + 1 trackID each)
        Returns:
            - matches: Array of id pairs (one per pair of matched showers)
            - vertices: Array of the corresponding pi0 decay vertices
        '''
        self.verbose = verbose
        
        sh_starts     = []
        sh_directions = []
        sh_energies   = []
        for sh_index, sh in enumerate(reco_showers):
            sh_starts.append(sh.start)
            sh_directions.append(sh.direction)
            sh_energies.append(sh.energy)
        
        if method == 'proximity':
            # Mask out points that are not track-like, return if there is no such point
            track_mask = np.where(segment[:,-1] < 2)[0]
            if not len(track_mask):
                return []
            track_data = segment[track_mask]

            # Get the best valid pairs (closest and compatible)
            tolerance = 10. # defines the max. allowed dist [px] between two lines of the shower dir. at their closest point of approach.
            matches, vertices, dists = self.find_best_pairs(np.array(sh_starts), np.array(sh_directions), tolerance)

            # Find the closest track-like point for each pair, adjust vertex position
            if len(vertices):
                vertices = self.find_closest_points(np.array(vertices), track_data[:,:3])

            return matches, vertices, dists
        
        elif method == 'ppn':
            matches, vertices = [], []
            
            # Return if < 2 reconstructed showers
            if len(ppn_track_points[0])<2 or len(sh_starts)<2:
                return matches, vertices
            
            # For all showers: Find pair-wise CPAs and closest distance of a shower's direction to the CPAs
            energy_threshold        = 15. # minimum energy [MeV] needed for showers to be taken into account in the search for pair-wise CPAs
            tolerance_CPA_to_shower = 30. # defines the max. dist. allowed of a CPA to the closest point on the shower's direction axis
            CPAs = self.find_pair_wise_CPAs(np.array(sh_energies),\
                                            np.array(sh_starts),\
                                            np.array(sh_directions),\
                                            tolerance_CPA_to_shower,\
                                            energy_threshold)
            
            # Check if there is at least 1 reasonable CPA
            n_CPAs = 0
            for i in range(len(sh_energies)):
                for j in range(len(sh_energies)):
                    if np.any(CPAs[i][j] == -float('inf')):
                        continue
                    else:
                        #print(' Accepted CPA: ', CPAs[i][j])
                        n_CPAs += 1
            #print(' Number of accepted CPAs:      ', n_CPAs)
            if n_CPAs < 1:
                return matches, vertices
            
            # For all CPAs: Find track-labeled PPN points close to a CPAs
            tolerance_CPA_to_PPN = 40. # defines the max. allowed distance of a CPA to the closest track-labeled PPN point
                                       # TODO: READ FROM CONFIG FILE, organize all tolerance parameters
            ppns = []
            for point in ppn_track_points[0]:
                ppns.append([point.ppns[0], point.ppns[1], point.ppns[2]])
            
            ppn_candidates = self.find_PPNs_close_to_CPAs(CPAs, ppns, tolerance_CPA_to_PPN)
            if len(ppn_candidates)==0:
                return matches, vertices
            #print(' ppn_candidates (close to CPAs): ', ppn_candidates)
            
            # Check if there is at least 1 PPN candidate which could be a pi0 decay vertex
            n_ppn_candidates = 0
            for i in range(len(sh_energies)):
                for j in range(len(sh_energies)):
                    if len(ppn_candidates[i][j]) > 0:
                        #print(' Found', len(ppn_candidates[i][j]), 'PPN candidates for showers', i, 'and', j, ': ')
                        #for cand in range(len(ppn_candidates[i][j])):
                        #    print('                     ', ppn_candidates[i][j][cand])
                        n_ppn_candidates += 1
            if n_ppn_candidates < 1:
                return matches, vertices
            
            # Veto showers which have a track-labeled PPN point close (< min_distance) to reco shower start (reject electrons/positrons)
            min_distance = 0 # reject showers starting close (< min_distance) to a vertex candidate (likely to be electron/positron induced showers)
            ppn_candidates = self.reject_electronic_showers(np.array(sh_starts), ppn_candidates, min_distance)
            # TODO: Maybe also consider: If > NUMBER track-labeled edeps are within RADIUS around shower_start: Reject?
            # Note: Sometimes, semantic segmentation makes mistake at the very beginning of a shower...
            #print(' ppn_candidates (electrons rejected): ', ppn_candidates)

            # Find showers with a direction approximatively (shower_start - vertex_candidate), 
            # reject showers starting too close (< min_distance) to a vertex candidate (e.g. from electrons)
            spatial_tolerance = 20. # if a sh_start's distance to a vertex_candidate is < spatial_tolerance : take the candidate as sh_start
                                    # TODO: Read from config file?
                                    # TODO: Optimise this value (it's probably too small)
            angular_tolerance = 20. # defines the maximum allowed difference between sh_direction and (sh_start-vertex_candidate) [in degrees]
                                    # TODO: Read from config file?
                                    # TODO: Optimise this value (it's probably too small)
        
            # Find vertices
            vertex_candidates, angle_matrix = self.find_vertex_candidates(np.array(sh_starts),\
                                                                          np.array(sh_directions),\
                                                                          ppn_candidates,\
                                                                          angular_tolerance)
            #print(' vertex_candidates: ', vertex_candidates)
            #print(' angle_matrix: ', angle_matrix)
            
            # Match shower pairs to pi0 decays
            max_angle_sum = 40. # defining the max. allowed sum of both angle-differences (between sh_i_direction and sh_i_start-vtx_candidate)
                                # in order to be accepted as a match
                                # TODO: Read from config file?
                                # TODO: Optimise this value (it's probably too large)
            matches, vertices = self.find_photon_pairs(vertex_candidates, angle_matrix, max_angle_sum)
            #print(' matches: ', matches)
            #print(' vertices: ', vertices)

            # Return
            return matches, vertices

        else:
            raise ValueError('Shower matching method not recognized:', method)


    def find_best_pairs(self, sh_starts, sh_directions, tolerance):
        '''
        Finds the indices of the best pairs, compute the interaction vertex for each pair
        Inputs:
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - tolerance: defines the max. allowed dist [px] between two lines of the shower dir. at their closest point of approach
        Returns:
            - Array of id pairs (one per pair of matched showers)
            - Array of production vertices
        '''
        
        # Get vertex (mean of CPAs) and the distance for each pair of showers, build a separation matrix
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
        Calculates the interaction vertex as the mean of the two CPAs
        and the minimum distance as the seperation between them.
        
        Inputs:
            - sh_starts (2 x 3): array of shower start points
            - sh_directions (2 x 3): array of shower directions
        Returns:
            - Interaction point
            - Shortest distance between the two lines
        """
        CPAs = self.find_CPAs(sh_starts, sh_directions)
        return np.mean(CPAs,axis=0), np.linalg.norm(CPAs[1]-CPAs[0])


    @staticmethod
    def find_CPAs(sh_starts, sh_directions):
        '''
        Calculates closest point of approach (CPA) between two vectors
        *in the backward direction*.
        If lines are parallel or closest point of approach is in the "forward" direction.
        Returns the separation between the two sh_starts.
        
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

        # Check for parallel lines, CPA undefined if parrallel
        if abs(v_dp) == 1:
            raise ValueError('The two shower axes are parallel, cannot find CPA')

        # Minimize the distance
        s0 = (-np.dot(d,v0) + np.dot(d,v1)*v_dp)/(1-v_dp**2)
        s1 = ( np.dot(d,v1) - np.dot(d,v0)*v_dp)/(1-v_dp**2)

        # Check that we have propogated both vectors in the backward dir
        if s0 > 0 or s1 > 0:
            return [sh_starts[0], sh_starts[1]]

        # Minimum separation
        CPA0 = sh_starts[0] + s0 * sh_directions[0]
        CPA1 = sh_starts[1] + s1 * sh_directions[1]
        return [CPA0, CPA1]


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


    def find_pair_wise_CPAs(self, sh_energies, sh_starts, sh_directions, tolerance_CPA_to_shower, energy_threshold):
        '''
        Finds the indices of the best pairs, compute the interaction vertex for each pair
        Inputs:
            - sh_energies (M x 3): array of shower energies, M corresponding to the number of reco showers in the event
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - tolerance_CPA_to_shower: defines the max. dist. allowed of a CPA to the closest point on the shower's direction axis
            - energy_threshold: min. energy deposit of a shower needed to look for CPAs with other showers
        Returns:
            - triangular (M x M)-matrix where the elements are the CPA coordinates between pairs of reco showers
        '''
        if self.verbose:
            print(' ------- in function find_pair_wise_CPAs -------')
            print(' sh_energies:              ', sh_energies)
            print(' sh_starts:                ', sh_starts)
            print(' sh_directions:            ', sh_directions)
            print(' tolerance_CPA_to_shower:  ', tolerance_CPA_to_shower)
            print(' energy_threshold:         ', energy_threshold)
        
        # Get CPAs between all shower pairs (and the distance between the two lines of the shower directions)
        n_sh = len(sh_energies)
        #print(' n_sh: ', n_sh)
        
        # Create (MxM)-matrix [where M = number of reco showers].
        # Each matrix element will be a list containing 3D space points with closest points of approach
        CPAs = np.zeros((n_sh,n_sh), dtype=object)
        
        # Loop over all matrix elements
        for i in range(n_sh):
            for j in range(n_sh):
                
                # CPAs matrix is upper/right triangular matrix -> continue if not in upper/right trinagle
                if j <= i:
                    CPAs[i,j] = np.array([-float('inf'),-float('inf'),-float('inf')])
                    continue
            
                # Continue if the shower has too little energy deposited
                if sh_energies[i] < energy_threshold or sh_energies[j] < energy_threshold:
                    CPAs[i,j] = np.array([-float('inf'),-float('inf'),-float('inf')])
                    continue

                # Get CPAs between all remaining shower pairs (and the distance between the two lines of the shower directions)
                P1 = sh_starts[i]
                P2 = sh_starts[j]
                dir_P1 = sh_directions[i]
                dir_P2 = sh_directions[j]

                # Get the (cos of the) angle between the direction vectors
                angle = np.dot(dir_P1,dir_P2) / np.linalg.norm(dir_P1)/np.linalg.norm(dir_P2)
                #print(' angle [deg]:          ', np.arccos(angle)*180/np.pi)
                
                # Check for parallel lines or angles which do not make sense (CPA would be undefined)
                if abs(angle) == 1 or abs(angle) > 1.:
                    print(' WARINING: The two shower axes are parallel or the angle does not make sense, cannot find CPA. ')
                    raise ValueError(' ValueError ... ')

                vec_P2_P1 = np.array(sh_starts[j])-np.array(sh_starts[i])
                #print(' vec_P2_P1: ', vec_P2_P1)

                # Vector orthogonal to both direction vectors (and its squared length)
                M = np.cross(np.array(sh_directions[j]), np.array(sh_directions[i]))
                #print(' M: ', M)
                m2 = np.dot(M,M)
                #print(' m2: ', m2)

                R = np.cross(vec_P2_P1, M)/m2
                #print(' R: ', R)

                tP1 = np.dot(R, sh_directions[j])
                tP2 = np.dot(R, sh_directions[i])
                #print(' tP1: ', tP1)
                #print(' tP2: ', tP2)

                # Points on direction lines closest to the CPA
                p1 = np.array(P1) + np.array(tP1) * np.array(dir_P1)
                p2 = np.array(sh_starts[j]) + np.array(tP2) * np.array(sh_directions[j])
                #print(' p1: ', p1)
                #print(' p2: ', p2)

                CPA = (np.array(p2) + np.array(p1)) / 2.
                #print(' CPA: ', CPA)

                # Shortest distance between the two shower's direction lines
                dist_between_lines = np.linalg.norm( np.dot(vec_P2_P1,M)) / (np.sqrt(m2))
                #print(' dist_between_lines: ', dist_between_lines)

                # Distance from closest point on line to CPA
                dist_to_CPA = np.linalg.norm( np.dot(vec_P2_P1,M)) / (np.sqrt(m2)) / 2.
                #print(' dist_to_CPA: ', dist_to_CPA)

                if dist_between_lines < tolerance_CPA_to_shower:
                    CPAs[i,j] = CPA
                else:
                    CPAs[i,j] = np.array([-float('inf'),-float('inf'),-float('inf')])
        #if self.verbose:  
            #print(' CPAs: ', CPAs)
        return np.array(CPAs)

    
    def find_PPNs_close_to_CPAs(self, CPAs, ppns, tolerance_CPA_to_PPN):
        '''
        Finds ppn_candidates = track-labeled PPN points which are close (distance < tolerance) to any CPA
        Inputs:
            - CPAs (M x M): matrix for pair-wise CPA. Every matrix entry is a np.array with the 3D coordinates of the CPA.
            - ppns (M x 3): array of track-labeled points from PPN (3D coordinates)
            - tolerance_CPA_to_PPN: float, defines the max. allowed distance of a CPA to the closest track-labeled PPN point
        Returns:
            - (M x M)-matrix where the elements are np.arrays of PPN points
        '''
        if self.verbose:
            print(' ------- in function find_PPNs_close_to_CPAs -------')
            #print(' CPAs:                 ', CPAs)
            print(' ppns:                 ', ppns)
            print(' tolerance_CPA_to_PPN: ', tolerance_CPA_to_PPN)

        # Check that CPAs matrix is quadratic
        assert np.shape(CPAs)[0] == np.shape(CPAs)[1]

        # Create matrix for PPN candidates
        ppn_candidates = np.zeros((np.shape(CPAs)[0],np.shape(CPAs)[1]), dtype=object)

        # Loop over matrix which contains the CPAs
        for i in range(np.shape(CPAs)[0]):
            for j in range(np.shape(CPAs)[1]):
                if j <= i:
                    ppn_candidates[i,j] = []
                    continue
                if np.any(CPAs[i][j] == -float('inf')):
                    ppn_candidates[i,j] = []
                    continue
                else:
                    #print(' Accepted CPA: ', CPAs[i][j])
                    accepted_PPNs = []
                    
                    # Loop over all PPN points and select those close (<tolerance_CPA_to_PPN) to the CPA
                    for ppn in ppns:
                        dist = np.linalg.norm(CPAs[i][j]-ppn)
                        if dist < tolerance_CPA_to_PPN:
                            accepted_PPNs.append(ppn)
                            #print(' Accepted PPN: ', ppn)

                    ppn_candidates[i,j] = accepted_PPNs
        if self.verbose:
            print(' ppn_candidates: ')
            print(ppn_candidates)

        return ppn_candidates
    
    
    def reject_electronic_showers(self, sh_starts, ppn_candidates, min_distance):
        '''
        Reject showers which have a reconstructed start point close (< 3 px) to at least one corresponding PPN candidate.
        This is done in order to reject electon/positron induced showers.
        Inputs:
            - sh_starts (M x 3): array of shower start points
            - ppn_candidates (M x M)-matrix where each element is a list containing 3D space points for every
              track-labeled PPN point close to the CPA of shower pairs
            - min_distance: float, defines the min. distance of a shower's start point to any correpsonding
              track-labeled PPN point to not be rejected
        Returns:
            - updated (M x M)-matrix where the elements are np.arrays of PPN points
        '''
        if self.verbose:
            print(' ------- in function reject_electronic_showers -------')
            print(' sh_starts:           ', sh_starts)
            print(' ppn_candidates:      ', ppn_candidates)
            print(' min_distance:        ', min_distance)
        
        for shower_1 in range(len(sh_starts)): #range(np.shape(ppn_candidates)[0]):
            for shower_2 in range(shower_1+1, len(sh_starts)): #range(shower_1+1, np.shape(ppn_candidates)[1]):
                for vtx_cand in ppn_candidates[shower_1][shower_2]:
                    dist_1 = np.linalg.norm(sh_starts[shower_1]-vtx_cand)
                    dist_2 = np.linalg.norm(sh_starts[shower_2]-vtx_cand)
                    if dist_1<min_distance or dist_2<min_distance:
                        if self.verbose:
                            print(' reject because shower start is too close to the ppn point: dist_1 =', dist_1, ' \t dist_2 =', dist_2)
                        ppn_candidates[shower_1][shower_2] = []
        
        return ppn_candidates

    
    def find_vertex_candidates(self, sh_starts, sh_directions, ppn_candidates, angular_tolerance):
        '''
        Finds vertex_candidates = track-labeled PPN points (close to the CPA)
        Inputs:
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - ppn_candidates (M x M): matrix where the elements are lists of np.arrays of PPN points (possible vertex candidates)
            - angular_tolerance: defines the maximum allowed difference between sh_direction and (sh_start-vertex_candidate) [in degrees]
        Returns:
            - (M,M)-matrix where each elements is an np.array of the 3D vertex position
            - (M,M)-matrix where each elements is the angular agreement of the shower's directions and the vertex candidate
        '''
        if self.verbose:
            print(' ------- in function find_vertex_candidates -------')
            print(' sh_starts:           ', sh_starts)
            print(' sh_directions:       ', sh_directions)
            print(' ppn_candidates:      ', ppn_candidates)
            print(' angular_tolerance:   ', angular_tolerance)
       
        # Get matrix with angular agreements
        angular_agreement = np.zeros((len(sh_starts),len(sh_starts)), dtype=object)
        
        # Check that angular_agreement matrix is quadratic (MxM) with M == number of reco showers
        assert np.shape(angular_agreement)[0] == np.shape(angular_agreement)[1]
        assert np.shape(angular_agreement)[0] == len(sh_starts)
        
        # Fill angular_agreement matrix
        for shower_1 in range(np.shape(angular_agreement)[0]):
            
            # Continue if shower_1 does not have an allocated direction
            if np.linalg.norm(sh_directions[shower_1]) == 0.:
                print(' WARNING: direction of shower', shower_1, 'not allocated... ')
                for shower_2 in range(np.shape(angular_agreement)[1]):
                        angular_agreement[shower_1][shower_2] = []
                continue
            
            for shower_2 in range(np.shape(angular_agreement)[1]):
                
                # angular_agreement is triangular -> continue if not in upper right triangle
                if shower_2 <= shower_1:
                    angular_agreement[shower_1,shower_2] = []
                    continue
                
                # Continue if shower_2 does not have an allocated direction
                if np.linalg.norm(sh_directions[shower_2]) == 0.:
                    print(' WARNING: direction of shower', shower_2, 'not allocated... ')
                    angular_agreement[shower_1,shower_2] = []
                    continue
                
                if len(ppn_candidates[shower_1][shower_2]) > 0:
                    agreement = []
                    for vtx_cand in ppn_candidates[shower_1][shower_2]:
                        
                        # Obtain angle [degrees] between sh_direction and sh_start-vtx_candidate_position
                        if abs(np.linalg.norm(sh_directions[shower_1])*np.linalg.norm(sh_starts[shower_1]-vtx_cand)) > 0. and \
                           abs(np.linalg.norm(sh_directions[shower_2])*np.linalg.norm(sh_starts[shower_2]-vtx_cand)) > 0.:
                            angle_1 = np.arccos(np.dot(sh_directions[shower_1],sh_starts[shower_1]-vtx_cand)\
                                               /(np.linalg.norm(sh_directions[shower_1])*np.linalg.norm(sh_starts[shower_1]-vtx_cand)))*180./(np.pi)
                            angle_2 = np.arccos(np.dot(sh_directions[shower_2],sh_starts[shower_2]-vtx_cand)\
                                               /(np.linalg.norm(sh_directions[shower_2])*np.linalg.norm(sh_starts[shower_2]-vtx_cand)))*180./(np.pi)
                        else:
                            angle_1 = 0.
                            angle_2 = 0.
                        # Reject if one of the angles is larger than the angular_tolerance
                        if angle_1>angular_tolerance or angle_2>angular_tolerance:
                            #print(' INFO: angle_1', angle_1, '>', angular_tolerance, '(shower', shower_1,\
                            #                ') or', angle_2, '>', angular_tolerance, '(shower', shower_2,\
                            #                '), vtx_cand:', vtx_cand, ' --> continue ...')
                            agreement.append([float('inf'),float('inf')])
                        else:
                            agreement.append([angle_1,angle_2])
                    angular_agreement[shower_1,shower_2] = agreement
                else:
                    angular_agreement[shower_1,shower_2] = []
        if self.verbose:
            print('angular_agreement', angular_agreement)
        
        # For every shower-pair, only keep the vtx_candidate with the best angular_agreement
        vertex_candidates = np.zeros((len(sh_starts),len(sh_starts)), dtype=object)
        angle_matrix      = np.zeros((len(sh_starts),len(sh_starts)), dtype=object)
        
        # TODO: Make shorter: min_angle_pair = angle_pair for i in angle_pairs where i[0]+i[1] is minimum
        for i in range(np.shape(angular_agreement)[0]):
            for j in range(np.shape(angular_agreement)[1]):
                min_angle_pair = float('inf')
                min_pair_index = -1
                for pair_index, angle_pair in enumerate(angular_agreement[i][j]):
                    if self.verbose:
                        print(' pair_index: ', pair_index)
                        print(' angle_pair: ', angle_pair)
                    summed_angles = float('inf')
                    if angle_pair[0] != -float('inf') and angle_pair[1] != -float('inf'):
                        summed_angles = angle_pair[0] + angle_pair[1]
                        if summed_angles < min_angle_pair:
                            min_pair_index = pair_index
                            min_angle_pair = summed_angles
                if min_angle_pair < (2.*angular_tolerance) and min_pair_index >= 0:
                    vertex_candidates[i][j] = ppn_candidates[i][j][min_pair_index]
                    angle_matrix[i][j] = angular_agreement[i][j][min_pair_index]
                else:
                    vertex_candidates[i][j] = []
                    angle_matrix[i][j] = []

        return vertex_candidates, angle_matrix


    def find_photon_pairs(self, vertex_candidates, angle_matrix, max_angle_sum):
        '''
        Finds matches (= pairs of showers originated from the same vertex)
        Inputs:
            - vertex_candidates (M x 3): array of vertex candidates positions, M = number of vertex candidates, 3 = x,y,z coordinates
            - angle_matrix: (M x M)-matrix where each elements is pair corresponding to the angular agreement of the
              corresponding shower-pair directions and the vertex candidate
            - max_angle_sum: float, defining the max. allowed sum of both angle-differences in order to be accepted as a match
        Returns:
            - matches: Array of shower_id pairs (one per pair of matched showers)
        '''
        if self.verbose:
            print(' ------- in function find_photon_pairs -------')
            print(' vertex_candidates: ', vertex_candidates)
            print(' angle_matrix:      ', angle_matrix)
        
        angle_sum_matrix = np.full((np.shape(angle_matrix)[0],np.shape(angle_matrix)[1]), float('inf'))
        
        for i in range(np.shape(angle_matrix)[0]):
            for j in range(np.shape(angle_matrix)[1]):
                #print(angle_matrix[i][j])
                if len(angle_matrix[i][j]) > 0:
                    angle_sum_matrix[i][j] = angle_matrix[i][j][0] + angle_matrix[i][j][1]
            if self.verbose:
                print(' angle_sum_matrix: ')
                print(angle_sum_matrix)
        
        matches = []
        pi0_vertices = []
        
        # Select smallest value from angle_sum_matrix and make all entries (rows and columns) of used
        # showers to float('inf') in order to not use the matched showers again
        for i in range(min(angle_sum_matrix.shape)):
            if angle_sum_matrix.min() < max_angle_sum:
                indices = np.argwhere(angle_sum_matrix.min() == angle_sum_matrix)
                matches.append([indices[0][0],indices[0][1]])
                pi0_vertices.append(vertex_candidates[indices[0][0]][indices[0][1]])
                #print(' indices: ', indices)
                #print(' matches: ', matches)
                #print(' pi0_vertices: ', pi0_vertices)
                angle_sum_matrix[indices[0][0],:] = float('inf')
                angle_sum_matrix[:,indices[0][0]] = float('inf')
                angle_sum_matrix[indices[0][1],:] = float('inf')
                angle_sum_matrix[:,indices[0][1]] = float('inf')
        if self.verbose:
            print(' matches:      ', matches)
            print(' pi0_vertices: ', pi0_vertices)

        return matches, pi0_vertices