from mlreco.main_funcs import process_config, train, inference
import yaml
from mlreco.main_funcs import process_config
from mlreco.iotools.factories import loader_factory
import numpy as np

import torch
from mlreco.iotools.factories import loader_factory
from mlreco.iotools.samplers import RandomSequenceSampler

from mlreco.utils.gnn.cluster import form_clusters_new, get_cluster_batch, get_cluster_label
from mlreco.utils.gnn.primary import assign_primaries3, assign_primaries
from mlreco.utils.gnn.network import primary_bipartite_incidence
from mlreco.utils.gnn.compton import filter_compton
from mlreco.utils.gnn.data import cluster_vtx_features, cluster_edge_features, edge_assignment
from mlreco.utils.groups import process_group_data

from mlreco.models.attention_gnn import BasicAttentionModel
from mlreco.utils.gnn.evaluation import assign_clusters
from mlreco.utils.data_parallel import DataParallel

def find_shower_gnn(dbscan, groups, em_primaries, energy_data, types, model_name, model_checkpoint, gpu_ind=0, verbose=False):
    """
    NOTE: THIS IS PROBABLY BROKEN; it was written right after the first pi0 workshop
    
    dbscan: data parsed from "dbscan_label": ["parse_dbscan", "sparse3d_fivetypes"]
    groups: data parsed from "group_label": ["parse_cluster3d_clean", "cluster3d_mcst", "sparse3d_fivetypes"]
    em_primaries: data parsed from "em_primaries" : ["parse_em_primaries", "sparse3d_data", "particle_mcst"]
    energy_data: data parsed from "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    
    returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary
    """
    event_data = [torch.tensor(dbscan), torch.tensor(em_primaries)]
    torch.cuda.set_device(0)
    model_attn = DataParallel(BasicAttentionModel(model_name),
                              device_ids=[0],
                              dense=False)
    
    model_attn.load_state_dict(torch.load(model_checkpoint, map_location='cuda:'+str(gpu_ind))['state_dict'])
    model_attn.eval().cuda()
    
    
    data_grp = process_group_data(torch.tensor(groups), torch.tensor(dbscan))
    
    clusts = form_clusters_new(dbscan)
    selection = filter_compton(clusts) # non-compton looking clusters
    clusts = clusts[selection]
    full_primaries = np.array(assign_primaries3(em_primaries, clusts, groups))
    primaries = assign_primaries(torch.tensor(em_primaries), clusts, torch.tensor(groups))
    batch = get_cluster_batch(dbscan, clusts)
    edge_index = primary_bipartite_incidence(batch, primaries, cuda=True)
    
    if len(edge_index) == 0: # no secondary clusters
        selected_voxels = []
        for p in full_primaries.astype(int):
            if p == -1:
                selected_voxels.append(np.array([]))
            else:
                selected_voxels.append(clusts[p])
        return selected_voxels
    
    n = len(clusts)
    mask = np.array([(i not in primaries) for i in range(n)])
    others = np.arange(n)[mask]
    
    pred_labels = model_attn(event_data)
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)

    count = 0
    selected_voxels = []
    for i in range(len(full_primaries)):
        p = full_primaries[i]
        if p == -1:
            selected_voxels.append(np.array([]))
        else:
            selected_clusts = clusts[np.where(pred_nodes == p)[0]]
            selected_voxels.append(np.concatenate(selected_clusts))
            
    return selected_voxels

def find_shower_gnn_with_cone(dbscan, groups, em_primaries, energy_data, types, model_name, model_checkpoint, gpu_ind=0, verbose=False):
    """
    dbscan: data parsed from "dbscan_label": ["parse_dbscan", "sparse3d_fivetypes"]
    groups: data parsed from "group_label": ["parse_cluster3d_clean", "cluster3d_mcst", "sparse3d_fivetypes"]
    em_primaries: data parsed from "em_primaries" : ["parse_em_primaries", "sparse3d_data", "particle_mcst"]
    energy_data: data parsed from "input_data": ["parse_sparse3d_scn", "sparse3d_data"]
    
    returns a list of length len(em_primaries) containing np arrays, each of which contains the indices corresponding to the voxels in the cone of the corresponding EM primary
    """
    event_data = [torch.tensor(dbscan), torch.tensor(em_primaries)]
    torch.cuda.set_device(0)
    model_attn = DataParallel(BasicAttentionModel(model_name),
                              device_ids=[0],
                              dense=False)
    
    model_attn.load_state_dict(torch.load(model_checkpoint, map_location='cuda:'+str(gpu_ind))['state_dict'])
    model_attn.eval().cuda()
    
    
    data_grp = process_group_data(torch.tensor(groups), torch.tensor(dbscan))
    
    clusts = form_clusters_new(dbscan)
    selection = filter_compton(clusts) # non-compton looking clusters
    clusts = clusts[selection]
    full_primaries = np.array(assign_primaries3(em_primaries, clusts, groups))
    primaries = assign_primaries(torch.tensor(em_primaries), clusts, torch.tensor(groups))
    batch = get_cluster_batch(dbscan, clusts)
    edge_index = primary_bipartite_incidence(batch, primaries, cuda=True)
    
    if len(edge_index) == 0: # no secondary clusters
        selected_voxels = []
        for p in full_primaries.astype(int):
            if p == -1:
                selected_voxels.append(np.array([]))
            else:
                selected_voxels.append(clusts[p])
        return selected_voxels
    
    n = len(clusts)
    mask = np.array([(i not in primaries) for i in range(n)])
    others = np.arange(n)[mask]
    
    pred_labels = model_attn(event_data)
    pred_nodes = assign_clusters(edge_index, pred_labels, primaries, others, n)

    count = 0
    selected_voxels = []
    for i in range(len(full_primaries)):
        p = full_primaries[i]
        if p == -1:
            selected_voxels.append(np.array([]))
        else:
            selected_clusts = clusts[np.where(pred_nodes == p)[0]]
            selected_voxels.append(np.concatenate(selected_clusts))
            
    return selected_voxels