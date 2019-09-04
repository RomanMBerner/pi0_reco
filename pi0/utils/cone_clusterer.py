import numpy as np
from mlreco.utils.gnn.cluster import form_clusters_new
from mlreco.utils.gnn.compton import filter_compton
from mlreco.visualization.voxels import scatter_label
from mlreco.utils.gnn.primary import assign_primaries_unique
from mlreco.utils import metrics

# TODO make sure entire primary cluster is assigned even if it's outside the cone

def cluster_cones(input, primaries, return_truth=False):
    """
    Inputs:
        - input (dict): output from dataiterator, which contains
        NAME (str) : TENSOR (np.ndarray) items (except for index (list of list))

        NOTE: <input> must be ghost-point removed. Current function uses 
        the coordinates of <segment_label> by default. 

    Returns:
        - prediction ((N, 1) np.array): prediction cluster labels for current input.
        Returns None if some necessary conditions are not met (see inline comments). 
        - cone_params (list of tuple): list of cone parameters for all cones attached 
        to the em primary. 

    NOTE: For safety, use batch size 1.
    """
    # Check if input contains showers
    mask = input['segment_label'][:, -1] == 2
    if mask.shape[0] < 1:
        return None
    if return_truth:
        return input['group_label'][mask][:, -1]
    else:
        fit_shower, cone_params = find_shower_cone(
            input['dbscan_label'][mask],
            input['group_label'][mask],
            primaries,
            input['input_data'][mask],
            input['segment_label'][mask],
            return_truth=False,
            verbose=False
        )
        pred = -np.ones(input['input_data'][:, 4][mask].shape)
        for i, indices in enumerate(fit_shower):
            if not len(indices):
                continue
            pred[indices] = input['group_label'][:, 4][mask][indices]
        
        return pred, cone_params



def find_shower_cone(dbscan, groups, em_primaries, energy_data, types, length_factor=14.107334041, slope_percentile=52.94032412, slope_factor=5.86322059, return_truth=False, verbose=False):
    """
    dbscan: data parsed from "dbscan_label": ["parse_dbscan", "sparse3d_fivetypes"]
    groups: data parsed from "group_label": ["parse_cluster3d_clean", "cluster3d_mcst", "sparse3d_fivetypes"]
    em_primaries: data parsed from "em_primaries" : ["parse_em_primaries", "sparse3d_data", "particle_mcst"]
    energy_data: data parsed from "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    types: (???) Fivetypes label Tensor (N x 5)

    returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary
    """
    length_factor = params[0]
    slope_percentile = params[1]
    slope_factor = params[2]
    
    dbscan = DBSCAN(eps=params[3], min_samples=3).fit(positions).labels_.reshape(-1, 1)
    dbscan = np.concatenate((positions, np.zeros((len(positions), 1)), dbscan), axis=1)
    
    clusts = form_clusters_new(dbscan)
    assigned_primaries = assign_primaries_unique(
            em_primaries, clusts, groups, use_labels=True).astype(int)
    selected_voxels = []
    true_voxels = []
    cone_params_list = []
    for i in range(len(assigned_primaries)):
        if assigned_primaries[i] != -1:
            c = clusts[assigned_primaries[i]]

            if return_truth:
                group_ids = np.unique(groups[c][:, -1])
                type_id = -1
                for g in groups[c]:
                    for j in range(len(types)):
                        if np.array_equal(g[:3], types[j][:3]):
                            type_id = types[j][-1]
                            break
                    if type_id != -1:
                        break
                true_indices = np.where(np.logical_and(np.isin(groups[:, -1], group_ids), types[:, -1] >= 2))[0]
                true_voxels.append(true_indices)

            p = em_primaries[i]
            em_point = p[:3]

            # find primary cluster axis
            primary_points = dbscan[c][:, :3]
            primary_center = np.average(primary_points.T, axis=1)
            primary_axis = primary_center - em_point

            # find furthest particle from cone axis (???)
            # COMMENT: Maybe not the furthest particle? This seems to select the slope by percentile. 
            primary_length = np.linalg.norm(primary_axis)
            direction = primary_axis / primary_length
            axis_distances = np.linalg.norm(np.cross(primary_points-primary_center, primary_points-em_point), axis=1)/primary_length
            axis_projections = np.dot(primary_points - em_point, direction)
            primary_slope = np.percentile(axis_distances/axis_projections, slope_percentile)
            
            # define a cone around the primary axis
            cone_length = length_factor * primary_length
            cone_slope = slope_factor * primary_slope
            cone_vertex = em_point
            cone_axis = direction

            cone_params = (cone_length, cone_slope, cone_vertex, cone_axis)
            cone_params_list.append(cone_params)

            classified_indices = []
            # Should be able to vectorize operation. 
            for j in range(len(dbscan)):
                point = types[j]
                if point[-1] < 2:
                    # ??? Why not != 2?
                    continue
                coord = point[:3]
                axis_dist = np.dot(coord - em_point, cone_axis)
                if 0 <= axis_dist and axis_dist <= cone_length:
                    cone_radius = axis_dist * cone_slope
                    point_radius = np.linalg.norm(np.cross(coord-(em_point + cone_axis), coord-em_point))
                    if point_radius < cone_radius:
                        # point inside cone
                        classified_indices.append(j)
            classified_indices = np.array(classified_indices)
            selected_voxels.append(classified_indices)
        else:
            selected_voxels.append(np.array([]))

    if return_truth:
        return true_voxels, selected_voxels, cone_params_list
    else:
        return selected_voxels, cone_params_list
