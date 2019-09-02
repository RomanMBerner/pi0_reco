'''
This module contains methods associated with making a selection of two gamma originating from the same point

'''
import numpy as np

def calculate_sep(data_vec0, data_vec1):
    '''
    Calculates point of closest approach (POCA) between two vectors *in the backward direction*. If lines are parallel or point of closest approach is in the "forward" direction returns the separation between the two points.
    Args:
        data_vec0 - an 1x6 matrix containing first vector info (x,y,z,dx,dy,dz)
        data_vec1 - an 1x6 matrix containing second vector info (x,y,z,dx,dy,dz)
    
    NOTE: The vectors (dx,dy,dz) must be normalized to unit vectors. 

    Returns:
        scalar for projection along data_vec0 to reach POCA
        scalar for projection along data_vec1 to reach POCA
        separation at point of closes approach (in backwards direction)
    
    '''
    d = data_vec0[:3]-data_vec1[:3]
    v0, v1 = data_vec0[-3:], data_vec1[-3:]
    v_dp = np.dot(v0,v1)
    # check for parallel lines
    if v_dp**2 == 1:
        return 0, 0, np.linalg.norm(d)
    v_dp = np.dot(v0,v1)
    s0 = (-np.dot(d,v0) + np.dot(d,v1)*v_dp)/(1-v_dp**2)
    s1 = ( np.dot(d,v1) - np.dot(d,v0)*v_dp)/(1-v_dp**2)
    # check that we have propogated both vectors in the backward dir
    if s0 > 0 or s1 > 0:
        return 0, 0, np.linalg.norm(d)
    # minimum separation
    sep = np.sqrt(np.clip(np.linalg.norm(d)**2 + 2*np.dot(d,v0)*s0 - \
        2*np.dot(d,v1)*s1 - 2*np.dot(v1,v0)*s0*s1 + s0**2 + s1**2,0,None))
    return s0, s1, sep

def get_best_pair_mask(data_dir, maximum_sep, exclude=np.empty(0)):
    '''
    Finds the indexes of the best pair
    Args:
        - data_dir: an Nx8 matrix containing shower directions 
        associated 1:1 for vertexes (x,y,z,batch,valid,dx,dy,dz)
        - maximum_sep: a maximum separation for best pair, 
        otherwise returns empty array.
        - exclude: indexes to exclude from calculation (array)
    Returns:
        vector of length 2 of indexes of best pair
        NxN matrix of distances calculated (upper triangular matrix)
    
    '''
    n = len(data_dir)
    if n < 2:
        return np.empty(0), np.empty((n,n))
#     assoc_dir, nonassoc_dir = np.empty((2,5)), np.empty((n-2,5))
    pair_sep = np.full((n,n), maximum_sep)
    for idx0, dir0 in enumerate(data_dir):
        for idx1, dir1 in enumerate(data_dir):
            if idx0 >= idx1:
                continue
            if idx0 in exclude or idx1 in exclude:
                continue
            if not data_dir[idx0][4] or not data_dir[idx1][4]:
                continue
            s0, s1, sep = calculate_sep(dir0, dir1)
            pair_sep[idx0, idx1] = sep
    best_match = np.where(pair_sep == np.amin(pair_sep))
    return_pair_sep = np.triu(pair_sep, 1)
    if any(pair_sep[best_match] >= maximum_sep):
        return np.empty(0), return_pair_sep
#     print(return_pair_sep)
    return best_match, return_pair_sep

def do_iterative_selection(data_dir, maximum_sep=3.):
    '''
    Selects the best candidate pairs of vertexes with shower directions defined by a unit vector. Is performed iteratively until no more candidates are found.
    Args:
        data_dir - an Nx8 matrix containing shower directions associated 1:1 for vertexes (x,y,z,batch,valid,dx,dy,dz)
        maximum_sep - a maximum separation for best pairs
    Returns:
        Nx5 matrix containing (x,y,z,batch,pair number), pair number is 0 if non match, 1 if best match, 2 if second best match, ...
        NxN matrix containing backwards point of closest approach for each shower pair (upper triangular values
    
    '''
    n = len(data_dir)
    data_label = np.zeros((n,5))
    data_label[:,:4] = data_dir[:,:4]
    excluded = np.empty(0)
    curr_label = 1
    sep_matrix = np.empty((n,n))
    while True:
        best_pair, iter_sep_matrix = get_best_pair_mask(data_dir, maximum_sep=maximum_sep, exclude=excluded)
        if curr_label == 1:
            sep_matrix = iter_sep_matrix
        if not len(best_pair):
            break
        data_label[best_pair[0],4] = curr_label
        data_label[best_pair[1],4] = curr_label
        curr_label += 1
        excluded = np.append(excluded, best_pair)
    return data_label, sep_matrix
        
def find_POCA(paired_primaries):
    """
    Inputs:
        - paired_primaries: 2 x 8 array, representing a pair of 
        em_showers.

    Returns:
        - poca: point of closest approach, given by the mean of the
        points of closest approach.
    """

    directions = np.atleast_2d(paired_primaries[:, 3:6])
    directions = directions / np.linalg.norm(directions, axis=1).reshape(
        directions.shape[0], 1)
    data_vec = np.hstack((paired_primaries[:, 0:3], directions))
    s0, s1, sep = calculate_sep(data_vec[0], data_vec[1])
    poca1 = data_vec[0][0:3] + s0 * data_vec[0][3:6]
    poca2 = data_vec[1][0:3] + s1 * data_vec[0][3:6]
    poca = (poca1 + poca2) / 2
    return poca