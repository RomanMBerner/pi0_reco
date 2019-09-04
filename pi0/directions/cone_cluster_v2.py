import numpy as np
from mlreco.utils.gnn.cluster import form_clusters_new
from mlreco.utils.gnn.compton import filter_compton
from mlreco.visualization.voxels import scatter_label
from mlreco.utils.gnn.primary import assign_primaries_unique
from mlreco.utils import metrics

from sklearn.cluster import DBSCAN

def make_shower_frags(input, eps=6.0, min_samples=30, shower_label=2):
    """
    Cluster showers for initial identification of shower stem using DBSCAN.

    NOTE: Batch size should be one. 

    Inputs: 
        - input (dict): output from dataiterator.
        - eps (float): epsilon parameter value for DBSCAN
        - min_samples (int): minimum number of neighboring samples for a point to be
        considered as a core point, (parameter of DBSCAN).
        - shower_label (int/float): semantic label for showers. 

    Returns:
        Shower fragments, with labels assigned by DBSCAN clustering.
    """
    mask = input['segment_label'][:, -1] == shower_label
    mask = np.logical_and(mask, input['input_data'][:, -1] < 0.05)
    coords = input['input_data'][:, :3][mask]
    frag_labels = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(coords)
    return frag_labels


def assign_frags_to_primary():
    pass