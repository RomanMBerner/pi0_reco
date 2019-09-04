from scipy.spatial import distance_matrix
from sklearn.neighbors import RadiusNeighborsRegressor
import numpy as np

ADD_MISSING_ENERGY = False
BLUR_ENERGY = True

def process_data(data):
    """
    data: input_true, input_reco, ghost_label, group_label
    returns: input, output
        input: intersection between reco and true, labeled with reco charge depositions
        output: intersection between reco and true, labeled with adjusted energy depositions
    """
    input_true = data['input_true']
    input_reco = data['input_reco']
    segment_label = data['segment_label']
    group_label = data['group_label']
    
    chosen_indices = []
    chosen_reco_indices = []
    
    current_batch = 0
    current_batch_selection = np.where(input_true[:, -2] == current_batch)[0]
    current_input_true = input_true[current_batch_selection]
    for r in range(len(input_reco)):
        row = input_reco[r]
        b = row[-2]
        if b != current_batch:
            current_batch = b
            current_batch_selection = np.where(input_true[:, -2] == current_batch)[0]
        pos = row[:3]
        region_selection = np.where((current_input_true[:, 0] == pos[0]) & (current_input_true[:, 1] == pos[1]))[0]
        input_true_region = current_input_true[region_selection]
        for i in range(len(input_true_region)):
            row2 = input_true_region[i]
            pos2 = row2[:3]
            if np.array_equal(pos, pos2):
                chosen_indices.append(current_batch_selection[region_selection[i]])
                chosen_reco_indices.append(r)
                break
    
    if len(chosen_indices) == 0:
        return None
    
    chosen_indices = np.array(chosen_indices)
    chosen_reco_indices = np.array(chosen_reco_indices)
    
    lost_data = np.delete(input_true, chosen_indices, axis=0)
    found_data = input_true[chosen_indices]
    
    # find where the chosen indices are in the group data
    lost_group_data = -np.ones((len(lost_data), len(lost_data[0])))
    ungrouped_data = -np.ones((len(lost_data), len(lost_data[0])))
    found_group_data = -np.ones((len(found_data), len(found_data[0])))
    for i in range(len(lost_data)):
        row = lost_data[i]
        filter0 = group_label[np.where(group_label[:, -2] == row[-2])]
        filter1 = filter0[np.where(filter0[:, 0] == row[0])]
        filter2 = filter1[np.where(filter1[:, 1] == row[1])]
        filter3 = filter2[np.where(filter2[:, 2] == row[2])]
        if len(filter3) == 0:
            ungrouped_data[i] = row
        else:
            g = filter3[0]
            lost_group_data[i] = g
    for i in range(len(found_data)):
        row = found_data[i]
        filter0 = group_label[np.where(group_label[:, -2] == row[-2])]
        filter1 = filter0[np.where(filter0[:, 0] == row[0])]
        filter2 = filter1[np.where(filter1[:, 1] == row[1])]
        filter3 = filter2[np.where(filter2[:, 2] == row[2])]
        g = filter3[0]
        found_group_data[i] = g
    
#     lost_group_data = np.delete(group_label, chosen_indices, axis=0)
#     found_group_data = group_label[chosen_indices]
    
    if ADD_MISSING_ENERGY:
        batches = np.unique(input_true[:, 3])
        for b in batches:
            # nearest neighbor assignment within group
            found_groups = np.unique(found_group_data[np.where(found_group_data[:, 3] == b)][:, -1])
            lost_batch_mask = lost_group_data[:, 3] == b
            found_batch_mask = found_group_data[:, 3] == b
            for g in found_groups:
                lost_selection = np.where(lost_batch_mask & (lost_group_data[:, -1] == g))[0]
                found_selection = np.where(found_batch_mask & (found_group_data[:, -1] == g))[0]
                ldata = lost_data[lost_selection]
                fdata = found_data[found_selection]
                lost_positions = ldata[:, :3]
                found_positions = fdata[:, :3]
                distances = distance_matrix(lost_positions, found_positions)
                closest_points = np.argmin(distances, axis=1)
                closest_energies = ldata[:, -1]
                for i in range(len(closest_points)):
                    found_data[found_selection[closest_points[i]]][-1] += closest_energies[i]

            # associated ungrouped voxels with nearest voxels, regardless of group
            lost_ungrouped = np.where((ungrouped_data[:, 3] == b))[0]
            if len(lost_ungrouped) > 0:
                found_selection = np.where(found_batch_mask)[0]
                ldata = lost_data[lost_ungrouped]
                fdata = found_data[found_selection]
                lost_positions = ldata[:, :3]
                found_positions = fdata[:, :3]
                distances = distance_matrix(lost_positions, found_positions)
                closest_points = np.argmin(distances, axis=1)
                closest_energies = ldata[:, -1]
                for i in range(len(closest_points)):
                    found_data[found_selection[closest_points[i]]][-1] += closest_energies[i]
    
    if BLUR_ENERGY:
        blur_kernel = 3
        for g in np.unique(found_group_data[:, -1]):
            inds = np.where(found_group_data[:, -1] == g)
            selection = found_data[inds]
            total_energy = np.sum(selection[:, -1])

            coords = selection[:, :3]
            energies = selection[:, -1]
            neigh = RadiusNeighborsRegressor(radius=blur_kernel)
            neigh.fit(coords, energies)
            selection[:, -1] = neigh.predict(coords)
            selection[:, -1] *= total_energy / np.sum(selection[:, -1])
            found_data[inds, -1] = selection[:, -1]
        
    segment_indices = segment_label[chosen_indices, -1].astype(int)
    segment_one_hot = np.zeros((len(segment_indices), 5))
    segment_one_hot[np.arange(len(segment_indices)), segment_indices] = 1
    out = np.concatenate((input_reco[chosen_reco_indices], segment_one_hot, np.expand_dims(found_data[:, -1], axis=1)), axis=1)
    return np.array(out), found_group_data[:, -1]