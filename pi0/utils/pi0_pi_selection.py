'''
This module contains methods for making a pi+ track selection and (pi0 production vertex)

'''
import numpy as np

from pi0.utils import gamma2_selection
from pi0.utils import gamma_direction

def calculate_vertex_loc(gamma0_vec, gamma1_vec):
    '''
    Use the backwards point of closest approach to determine a potential pi0 production vertex
    Args:
        gamma0_vec - length 6 vector of (x,y,z,dx,dy,dz)
        gamma1_vec - length 6 vector of (x,y,z,dx,dy,dz)
    Return:
        length 4 vector of (x,y,z,l) of vertex, where l is the separation of the two gamma vectors at the vertex
    
    '''
    s0, s1, l = gamma2_selection.calculate_sep(gamma0_vec, gamma1_vec)
    x0 = gamma0_vec[:3] + s0 * gamma0_vec[-3:]
    x1 = gamma1_vec[:3] + s1 * gamma1_vec[-3:]
    vtx = np.mean([x0, x1],axis=0)
    return np.array([vtx[0], vtx[1], vtx[2], l])


def do_selection(label_data, gamma0_data, gamma1_data, mip_label=1, tolerance=10.):
    '''
    Project 2 paired gammas back to their crossing point and return nearest 
    (within tolerance) mip-track labeled hit as vertex candidate
    Args:
        label_data - Nx5 array with 5 being (x,y,z,batch,fivetypes label)
        gamma0_data - 1x7 array with 7 being (x,y,z,batch,dx,dy,dz)
        gamma1_data - 1x7 array with 7 being (x,y,z,batch,dx,dy,dz)
        mip_label - fivetypes particle label for mip track to associate
        tolerance - radius to search for a mip track label
    Return:
        length 5 array of (x,y,z,batch,l), l is the separation of the 
        two gammas at the proposed vertex
    '''
    track_mask = label_data[:,-1] == mip_label
    track_data = label_data[track_mask]
    if not len(track_data):
        return np.empty((0,5))
    vtx_info = calculate_vertex_loc(gamma0_data, gamma1_data)
    distances = gamma_direction.norm(track_data[:,:3], vtx_info[:3])
    min_dist_idx = np.argmin(distances)
    if distances[min_dist_idx] > tolerance:
        return np.empty((0,5))
    best_hit = track_data[min_dist_idx]
    return_arr = best_hit
    return_arr[4] = vtx_info[-1]
    return return_arr


def generate_pair_labels(data, em_primaries, predict=True):
    """
    Inputs:
        - 
    Returns:
        - pair_labels ((N, ) np.ndarray): labels of shape data['group_label']
        where the pairs obtain a unique label and all unpaired voxels receive "-1". 

        - poca_list (list of tuples): list of POCA coordinates for each paired clusters. 
    """
    # FIXME: Need to choose which method of estimating direciton
    # Also ignore PCA variables for the moment. 
    gamma_dir = gamma_direction.do_calculation(data['segment_label'], em_primaries)[0]
    if predict:
        primaries = gamma_dir.copy()
        primaries[:, [3,4,5,6,7]] = primaries[:, [5,6,7,3,4]]
    else:
        primaries = em_primaries.copy()
    selected_showers, sep_matrix = gamma2_selection.do_iterative_selection(
        gamma_dir, maximum_sep=5.)
    valid_mask = selected_showers[:, -1] != 0
    paired_list = np.unique(selected_showers[:, -1][valid_mask])
    pair_labels = -np.ones((data['group_label'].shape[0], ))
    poca_list = []
    if not len(paired_list):
        return None, []
    for i, plabel in enumerate(paired_list):
        shower_pair = primaries[selected_showers[:, -1] == plabel]
        print(shower_pair)
        poca = gamma2_selection.find_POCA(shower_pair)
        selection = do_selection(data['segment_label'], shower_pair[0], shower_pair[1])
        if selection.shape[0] < 1:
            continue
        poca_list.append(poca)
        paired_cluster_labels = em_primaries[selected_showers[:, -1] == plabel][:, -1]
        mask_pair = np.logical_or(data['group_label'][:, -1] == paired_cluster_labels[0],
                                  data['group_label'][:, -1] == paired_cluster_labels[1])
        pair_labels[mask_pair] = i+1
    return pair_labels, poca_list