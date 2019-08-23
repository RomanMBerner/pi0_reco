import numpy as np
from mlreco.utils.gnn.cluster import form_clusters_new
from mlreco.utils.gnn.compton import filter_compton
from mlreco.visualization.voxels import scatter_label
from mlreco.utils.gnn.primary import assign_primaries_unique
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier

def cluster(positions, em_primaries, params=[14.107334041, 52.94032412, 5.86322059, 1.01], inclusive=True):
    """
    positions: Nx3 array of EM shower voxel positions
    em_primaries: Nx3 array of EM primary positions
    
    if inclusive=True: returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary; note that each voxel might thus have multiple labels
    if inclusive=False: returns a tuple (arr of length len(em_primaries), arr of length len(positions)) corresponding to EM primary labels and the voxel labels; note that each voxel has a unique label
    """
    length_factor = params[0]
    slope_percentile = params[1]
    slope_factor = params[2]
    
    dbscan = DBSCAN(eps=params[3], min_samples=3).fit(positions).labels_.reshape(-1, 1)
    dbscan = np.concatenate((positions, np.zeros((len(positions), 1)), dbscan), axis=1)
    
    clusts = form_clusters_new(dbscan)
    selected_voxels = []
    true_voxels = []
    
    if len(clusts) == 0:
        # assignn everything to first primary
        selected_voxels.append(np.arange(len(dbscan)))
        print('all clusters identified as Compton')
        return selected_voxels
    assigned_primaries = assign_primaries_unique(np.concatenate((em_primaries, np.zeros((len(em_primaries), 2))), axis=1), clusts, np.concatenate((positions, np.zeros((len(positions), 2))), axis=1)).astype(int)
    for i in range(len(assigned_primaries)):
        if assigned_primaries[i] != -1:
            c = clusts[assigned_primaries[i]]
            
            p = em_primaries[i]
            em_point = p[:3]

            # find primary cluster axis
            primary_points = dbscan[c][:, :3]
            primary_center = np.average(primary_points.T, axis=1)
            primary_axis = primary_center - em_point

            # find furthest particle from cone axis
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

            classified_indices = []
            for j in range(len(dbscan)):
                point = positions[j]
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
    
    # don't require that each voxel can only be in one group
    if inclusive:
        return selected_voxels
    
    # require each voxel can only be in one group (order groups in descending size to overwrite large groups)
    em_primary_labels = -np.ones(len(selected_voxels))
    node_labels = -np.ones(len(positions))
    lengths = []
    for group in selected_voxels:
        lengths.append(len(group))
    sorter = np.argsort(lengths)[::-1]
    for l in range(len(selected_voxels)):
        if len(selected_voxels[sorter[l]]) > 0:
            node_labels[selected_voxels[sorter[l]]] = l
            em_primary_labels[sorter[l]] = l
    
    labeled = np.where(node_labels != -1)
    unlabeled = np.where(node_labels == -1)
    if len(labeled[0]) > 5 and len(unlabeled[0]) > 0:
        classified_positions = positions[labeled]
        unclassified_positions = positions[unlabeled]
        cl = KNeighborsClassifier(n_neighbors=2)
        cl.fit(classified_positions, node_labels[labeled])
        node_labels[unlabeled] = cl.predict(unclassified_positions)
    
    return em_primary_labels, node_labels