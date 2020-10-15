import numpy as np
from scipy.spatial import distance


class Pi0Matcher():

    def find_matches(self, reco_showers, segment, method, *ppn_track_points):
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
            - ppn_track_points (K x 3): array of track-labeled PPN points (K points, 3 coordinates each)
        Returns:
            - matches: Array of id pairs (one per pair of matched showers)
            - vertices: Array of the corresponding pi0 decay vertices
        '''

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
            if len(ppn_track_points[0])==0 or len(sh_starts)==0:
                return matches, vertices
            
            # For all showers: Find pair-wise CPAs and closest distance of a shower's direction to the CPAs
            energy_threshold = 15. # minimum energy [MeV] needed for showers to be taken into account in the search for pair-wise CPAs
            tolerance_CPA_to_shower = 20. # defines the max. dist. allowed of a CPA to the closest point on the shower's direction axis
            CPAs = self.find_pair_wise_CPAs(np.array(sh_energies), np.array(sh_starts), np.array(sh_directions), tolerance_CPA_to_shower, energy_threshold)
            print(' CPAs: ', CPAs)
            
            # For all CPAs: Find track-labeled PPN points close to a CPA
            ppns = []
            for point in ppn_track_points[0]:
                ppns.append([point.ppns[0], point.ppns[1], point.ppns[2]])
            
            tolerance_CPA_to_PPN = 40. # defines the max. allowed distance of a CPA to the closest track-labeled PPN point
            # TODO: READ FROM CONFIG FILE OR FEED IN find_matches_function // TODO: Better organize all tolerance parameters
            ppn_candidates = self.find_PPNs_close_to_CPAs(CPAs, ppns, tolerance_CPA_to_PPN)
            if len(ppn_candidates)==0:
                return matches, vertices
            print(' ppn_candidates: ', ppn_candidates)

            # Find showers with a direction approximatively (shower_start - vertex_candidate), 
            # reject showers starting too close (< min_distance) to a vertex candidate (e.g. from electrons)
            spatial_tolerance = 20. # if a sh_start's distance to a vertex_candidate is < spatial_tolerance : take the candidate as sh_start
                                    # TODO: Read from config file?
                                    # TODO: Optimise this value (it's probably too small)
            min_distance      = 3   # reject showers starting close (< min_distance) to a vertex candidate (liekly to be electron/positron induced showers)
            angular_tolerance = 20. # defines the maximum allowed difference between sh_direction and (sh_start-vertex_candidate) [in degrees]
                                    # TODO: Read from config file?
                                    # TODO: Optimise this value (it's probably too small)
        
            # Find vertex candidates
            vertex_candidates, reject_ind = self.find_vertex_candidates(np.array(sh_starts), np.array(sh_directions), np.array(sh_energies), ppn_candidates, spatial_tolerance, min_distance, angular_tolerance)
            if len(vertex_candidates)==0:
                return matches, vertex_candidates

            # Reject showers which have only very little deposited energy
            reject_ind = self.reject_low_energy_showers(np.array(sh_energies), energy_threshold, reject_ind)

            # Match shower pairs to pi0 decays
            matches, vertices = self.find_photon_pairs(vertex_candidates, reject_ind, sh_starts, sh_directions)

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
            - sh_energies (M x 3): array of shower energies
            - sh_starts (M x 3): array of shower start points
            - sh_directions (M x 3): array of shower directions
            - tolerance_CPA_to_shower: defines the max. dist. allowed of a CPA to the closest point on the shower's direction axis
            - energy_threshold: min. energy deposit of a shower needed to look for CPAs with other showers
        Returns:
            - Array of CPAs (possible pi0 vertices)
        '''
        #print(' ------- in function find_pair_wise_CPAs -------')
        #print(' sh_starts:                ', sh_starts)
        #print(' sh_directions:            ', sh_directions)
        #print(' tolerance_CPA_to_shower: ', tolerance_CPA_to_shower)
        
        # Get CPAs between all shower pairs (and the distance between the two lines of the shower directions)
        CPAs = []
        npoints = len(sh_starts)
        for i in range(npoints):

            # Continue if the shower has too little energy deposited
            if sh_energies[i] < energy_threshold:
                continue

            for j in range(i+1, npoints):

                # Continue if the shower has too little energy deposited
                if sh_energies[j] < energy_threshold:
                    continue

                #print(' i: ', i, ' \t j: ', j)

                # Get CPAs between all shower pairs (and the distance between the two lines of the shower directions)
                P1 = sh_starts[i]
                P2 = sh_starts[j]
                dir_P1 = sh_directions[i]
                dir_P2 = sh_directions[j]

                # Get the (cos of the) angle between the direction vectors
                angle = np.dot(dir_P1,dir_P2) / np.linalg.norm(dir_P1)/np.linalg.norm(dir_P2)
                #print(' angle [deg]:          ', np.arccos(angle)*180/np.pi)
                # Check for parallel lines or angles which do not make sense (CPA would be undefined)
                if abs(angle) == 1 or abs(angle) > 1.:
                    raise ValueError('The two shower axes are parallel or the angle does not make sense, cannot find CPA')

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
                    CPAs.append(CPA)

                '''
                # Get the (cos of the) angle between the direction vectors
                angle = np.dot(sh_directions[i], sh_directions[j])
                print(' angle:          ', angle)
                # Check for parallel lines or angles which do not make sense (CPA would be undefined)
                if abs(angle) == 1 or abs(angle) > 1.:
                    raise ValueError('The two shower axes are parallel or the angle does not make sense, cannot find CPA')

                dist = abs((sh_starts[j]-sh_starts[i])*angle)/angle
                print(' dist:           ', dist)

                CPA, dist = [1.,1.,1.], 1.
                #CPA, dist = self.find_vertex([sh_starts[i], sh_starts[j]], [sh_directions[i], sh_directions[j]])
                print(' CPA:            ', CPAs)
                print(' dist:           ', dist)
                if dist < tolerance_CPA_to_shower:
                    CPAs.append(CPA)
                '''
        #print(' CPAs:          ', CPAs)

        return np.array(CPAs)

    def find_PPNs_close_to_CPAs(self, CPAs, ppns, tolerance_CPA_to_PPN):
        '''
        Finds ppn_candidates = track-labeled PPN points which are close (distance < tolerance) to any CPA
        Inputs:
            - CPAs (N x 3): array of CPA points (3D coordinates)
            - ppns (M x 3): array of track-labeled points from PPN (3D coordinates)
            - tolerance_CPA_to_PPN: defines the max. allowed distance of a CPA to the closest track-labeled PPN point
        Returns:
            - Array of ppn_candidates (possible pi0 vertices)
        '''
        print(' ------- in function find_PPNs_close_to_CPAs -------')
        print(' CPAs:                 ', CPAs)
        print(' ppns:                  ', ppns)
        print(' tolerance_CPA_to_PPN: ', tolerance_CPA_to_PPN)
        
        distances = distance.cdist(CPAs,ppns)
        print(' distances:         ', distances)

        # Get the closest track_labeled PPN point for each CPA.
        # If the distance between this point and the CPA is < tolerance_CPA_to_PPN, return it as vertex_candidate
        smallest_idx = np.argmin(distance.cdist(CPAs,ppns),axis=1)
        print(' smallest_idx: ', smallest_idx)

        used_indices = []
        ppn_candidates = []
        
        for i, dist in enumerate(distances[0]):
            if dist < tolerance_CPA_to_PPN:
                used_indices.append(i)
                ppn_candidates.append(ppns[i])
        '''
        for i, idx in enumerate(idxs):
            if (distances[i][idx]<tolerance_CPA_to_PPN and idx not in used_indices):
                used_indices.append(idx)
                ppn_candidates.append(ppns[idx])
        '''
                
                
        print(' ppn_candidates: ', ppn_candidates)

        return np.array(ppn_candidates)


    def find_vertex_candidates(self, sh_starts, sh_directions, sh_energies, ppn_candidates, spatial_tolerance, min_distance, angular_tolerance):
        '''
        Finds vertex_candidates = track-labeled PPN points which are close (distance < tolerance) to any CPA
        Inputs:
            - sh_starts (N x 3): array of shower start points
            - sh_directions (N x 3): array of shower directions
            - sh_energies (N x 1): array of shower energies
            - ppn_candidates (M x 3): array of ppn candidates
            - spatial_tolerance: if a sh_start's distance to a vertex_candidate is < spatial_tolerance : take the candidate as sh_start
            - min_distance: reject showers starting close (< min_distance) to a vertex candidate (liekly to be electron/positron induced showers)
            - angular_tolerance: defines the maximum allowed difference between sh_direction and (sh_start-vertex_candidate) [in degrees]
        Returns:
            - Array of shower_vertices
        '''
        print(' ------- in function find_vertex_candidates -------')
        print(' sh_starts:           ', sh_starts)
        print(' sh_directions:       ', sh_directions)
        print(' ppn_candidates:   ', ppn_candidates)
        print(' spatial_tolerance:   ', spatial_tolerance)
        print(' min_distance:        ', min_distance)
        print(' angular_tolerance:   ', angular_tolerance)
        
        vertex_candidates = [[None, None, None] for _ in range(len(sh_starts))]

        # For all showers which have sh_start close (up to spatial_tolerance) to a vertex_candidate:
        # Take the vertex_candidate as the shower's vertex
        dist = distance.cdist(sh_starts,ppn_candidates)
        idxs = np.argmin(distance.cdist(sh_starts,ppn_candidates),axis=1)

        used_sh_indices     = []
        used_sh_vertices    = []
        rejected_sh_indices = []
        for i, idx in enumerate(idxs):
            # If shower_start is close (<spatial_tolerance) to vertex candidate, but not too close (>min_distance)
            # in order to reject electrons, take vertex_candidate as pi0 decay vertex.
            # TODO: Consider using the electron-photon separation (e- and photon-likelihood fractions)
            # -> reject shower if photon likelihood < MIN_VALUE or electron likelihood > MAX_VALUE
            if (dist[i][idx]<spatial_tolerance):
                if dist[i][idx]>min_distance:
                    used_sh_indices.append(i)
                    used_sh_vertices.append(ppn_candidates[idx])
                    #print(' Take vertex ', ppn_candidates[idx], ' since it is close to shower ', i, ' with start point ', sh_starts[i])
                else:
                    rejected_sh_indices.append(i)
                    #print(' rejected sh ', i)
        print(' rejected_sh_indices: ', rejected_sh_indices)

        # For all other showers which do not have a vertex_candidate close enough:
        # Take as vertex vertex_candidate which (in combination with sh_start) is in best angular agreement with the sh_direction
        counter = 0
        for sh_index, sh_start in enumerate(sh_starts):
            if sh_index in used_sh_indices:
                #vertex_candidates.append(used_sh_vertices[counter])
                vertex_candidates[sh_index] = used_sh_vertices[counter]
                counter += 1
            else:
                if sh_index in rejected_sh_indices:
                    continue
                else:
                    min_angle = float('inf')
                    for vtx_index, vtx_pos in enumerate(ppn_candidates):
                        angle = np.arccos(np.dot((sh_start-vtx_pos)/np.linalg.norm(sh_start-vtx_pos),sh_directions[sh_index]))*360./(2.*np.pi)
                        if angle < min_angle:
                            selected_vtx_index = vtx_index
                            selected_vtx_pos = vtx_pos
                            min_angle = angle
                    if min_angle >= angular_tolerance:
                        selected_vtx_index = None
                        selected_vtx_pos = np.array([None,None,None])
                    #vertex_candidates.append(selected_vtx_pos)
                    vertex_candidates[sh_index] = selected_vtx_pos
        print(' vertex_candidates:            ', vertex_candidates)

        return np.array(vertex_candidates), np.array(rejected_sh_indices)


    def reject_low_energy_showers(self, sh_energies, energy_threshold, reject_ind):
        '''
        Reject showers which have only very few edeps
        Inputs:
            - sh_energies (M x 1): array of shower energies
            - energy_threshold: min. energy deposit needed for a shower not to be rejected
            - reject_ind: array of shower indices which should be rejected (since they start too close to a vertex)
        Returns:
            - reject_ind: array of shower indices which should be rejected (since they start too close to a vertex or have to few edeps)
        '''
        #print(' ------- in function reject_low_energy_showers -------')
        #print(' sh_energies:      ', sh_energies)
        #print(' energy_threshold: ', energy_threshold)
        #print(' reject_ind:       ', reject_ind)

        for sh_index, sh_energy in enumerate(sh_energies):
            if sh_energy < energy_threshold:
                if sh_index not in reject_ind:
                    reject_ind = np.append(reject_ind, sh_index)
        #print(' reject_ind:    ', reject_ind)
        return np.array(reject_ind)


    def find_photon_pairs(self, vertex_candidates, reject_ind, sh_starts, sh_directions):
        '''
        Finds matches (= pairs of showers originated from the same vertex)
        Inputs:
            - vertex_candidates (M x 3): array of vertex candidates positions, M = number of vertex candidates, 3 = x,y,z coordinates
            - reject_ind: array of shower indices which should be rejected (since they start too close to a vertex candidate or have too few edeps)
            - sh_starts (N x 3): array of shower start points
            - sh_directions (N x 3): array of shower directions
        Returns:
            - matches: Array of shower_id pairs (one per pair of matched showers)
        '''
        print(' ------- in function find_photon_pairs -------')
        print(' vertex_candidates: ', vertex_candidates)
        print(' reject_ind:        ', reject_ind)
        print(' sh_starts:         ', sh_starts)
        print(' sh_directions:     ', sh_directions)
        
        # For every vertex candidate, check which shower indices agree (up to 20 degrees) with it
        max_angle = 20.
        
        sh_indices_list = [[] for _ in range(len(vertex_candidates))]
        angle_agreement_list = [[] for _ in range(len(vertex_candidates))]
        
        for vtx_ind, vtx in enumerate(vertex_candidates):
            for sh_ind, sh_dir in enumerate(sh_directions):
                print(' vtx: ', vtx)
                print(' sh_dir: ', sh_dir)
                print(' sh_starts: ', sh_starts[sh_ind])
                print(sh_starts[sh_ind]-vtx)
                print(np.linalg.norm(sh_starts[sh_ind]-vtx))
                print(sh_directions[sh_ind])
                angle = np.arccos(np.dot((sh_starts[sh_ind]-np.asarray(vtx))/np.linalg.norm(sh_starts[sh_ind]-np.asarray(vtx)),sh_directions[sh_ind]))*360./(2.*np.pi)
                print(' angle: ', angle)
                if  angle < max_angle:
                    sh_indices_list[vtx_ind].append(sh_ind)
                    angle_agreement_list[vtx_ind].append(angle)
        print(' sh_indices_list: ', sh_indices_list)
        print(' angle_agreement_list: ', angle_agreement_list)
                    

        # TODO: Could be interesting to close the kinematics for the cases where >2 showers originated from the same vertex.
        matches = []

        # Make set of vertex_candidates which occur more than once
        vtx_set = set(tuple(arr) for arr in vertex_candidates if not np.any(arr)==None)
        #print(' vtx_set: ', vtx_set)

        vtx_indices = [[] for _ in range(len(vtx_set))]

        for vtx_ind, vtx in enumerate(vtx_set):
            #print(' vtx: ', vtx)
            for vertex_ind, vertex in enumerate(vertex_candidates):
                #print(' vertex_ind: ', vertex_ind)
                #print(' vertex: ', vertex)
                if vertex_ind in reject_ind:
                    continue
                if np.array_equal(np.array(vtx),np.array(vertex)):
                    #print(' equal: ', np.array(vtx),np.array(vertex))
                    vtx_indices[vtx_ind].append(vertex_ind)
        #print(' vtx_indices: ', vtx_indices)

        for idx, ind in enumerate(vtx_indices):
            if len(ind) < 2:
                continue

            elif len(ind) == 2:
                matches.append(ind)

            elif len(ind) == 3:

                # Match those two showers which have the best angular agreement (= angle between sh_dir and (vtx_pos-sh_start))
                # TODO: Consider matching those showers which have the largest photon likelihood
                min_angle_1 = float('inf')
                min_angle_2 = float('inf')
                min_index_1 = -9
                min_index_2 = -8
                for counter, sh_ind in enumerate(ind):
                    angle = np.arccos(np.dot((sh_starts[sh_ind]-vertex_candidates[sh_ind])/np.linalg.norm(sh_starts[sh_ind]-\
                            vertex_candidates[sh_ind]),sh_directions[sh_ind]))*360./(2.*np.pi)
                    if angle < min_angle_1:
                        min_angle_2 = min_angle_1
                        min_index_2 = min_index_1
                        min_angle_1 = angle
                        min_index_1 = sh_ind
                    else:
                        if angle < min_angle_2:
                            min_angle_2 = angle
                            min_index_2 = sh_ind
                matches.append([min_index_1,min_index_2])

            else:
                print(' WARNING:', len(ind), 'showers pointing to the same vertex candidate -> Do not match any of the shower pairs to a pi0... ')

        print(' matches:       ', matches)

        pi0_vertices = []
        for i, match in enumerate(matches):
            pi0_vertices.append(vertex_candidates[match[0]])

        print(' reconstructed pi0_vertices: ', pi0_vertices)

        return matches, pi0_vertices