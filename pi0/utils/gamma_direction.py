'''
This module contains methods associated with inferring the direction of a gamma shower

''' 
import numpy as np
from sklearn.cluster import DBSCAN

def norm(xyz_hit, xyz_vtx):
    '''
    Calculates the euclidean distance from xyz_vtx to each hit in xyz_hit
    Args:
        xyz_hit - an Nx3 array of hit locations (x,y,z)
        xyz_vtx - a length 3 (x,y,z) location of the point to calculate distances from
    Return:
        an Nx1 array of distances for each hit location to the vtx
    '''
    xyz_d = xyz_hit[:,:3] - xyz_vtx
    distances = np.sqrt(np.sum(xyz_d**2, axis=1))
    return distances
#     return np.array([np.sqrt(xyz[0]**2 + xyz[1]**2 + xyz[2]**2) for xyz in xyz_d])

def dbscan_find_primary(xyz_hit, eps, min_samples):
    '''
    Performs DBSCAN on the specified hits
    Args:
        xyz_hit - Nx3 array of (x,y,z) of each hit
        eps - epsilon for dbscan
        min_samples - min_samples for dbscan
    Return:
        Nx1 mask for most common cluster, if not noise, else returns Nx1 array of True
    
    '''
    db = DBSCAN(eps=eps, min_samples=min_samples).fit(xyz_hit)
    core_hits_mask = np.zeros_like(db.labels_, dtype=bool)
    core_hits_mask[db.core_sample_indices_] = True
    labels = db.labels_
    (unique_labels, counts) = np.unique(labels[labels != -1], return_counts=True)
    max_label = -1
    if any(counts >= 0):
        max_label = unique_labels[np.argmax(counts)]
    if not max_label == -1:
        return labels == max_label & core_hits_mask
    return labels == max_label
    
def pca(xyz_hit):
    '''
    Performs a PCA fit to hit locations
    Args:
        xyz_hit - a Nx3 numpy array of hit locations
    Returns:
        2-``tuple`` of covariance matrix eigenvectors [3x3] and eigenvalues[3x1], sorted by decreasing eigenvalue
    '''
    cov = np.cov(xyz_hit, rowvar=False)
    evals, evecs = np.linalg.eigh(cov)
    return_order = np.argsort(evals)[::-1]
    return (evecs.T)[return_order], evals[return_order]

def compute_parity_flip(xyz_hit, vector, origin=(0,0,0)):
    '''
    Uses the dot product of the average xyz_hit vector and the specified vector to determine if vector in same, opposite, or parallel to most hits
    Args:
        xyz_hit - an Nx3 array of hit locations
        vector - a length 3 comparison vector
        origin - an optional offset to remove from each xyz_hit vector
    Returns:
        -1 if vector and avg xyz_hit vector are in opposite directions and +1 if in the same direction. 0 if they are perpendicular
    '''
    xyz_d = xyz_hit - origin
    xyz_avg = np.mean(xyz_d, axis=0)
    dot = np.dot(xyz_avg, vector)
    if dot > 0:
        return +1.
    elif dot < 0:
        return -1.
    return 0.

def do_calculation(shower_hits, vtx_data, radius=10., eps=2., min_samples=5, shower_label=2):
    '''
    Calculates the best fit direction of the gamma shower based on a PCA of the
    shower start
    Args:
        data - a numpy array Nx5 containing hits associated with gamma shower (x,y,z,batch,fivetypes label)
        vtxs - a numpy array Mx5 containing shower vertexes (x,y,z,batch,label)
        radius - hits within this radius are passed along to dbscan
        eps - epsilon for dbscan
        min_samples - min_samples for dbscan
        shower_label - label number for filtering ``data`` by fivetypes label
    Returns:
        a numpy array Mx8 containing (x,y,z,batch,valid,x_comp,y_comp,z_comp) across vertexes. XX_comp are the components of the unit vectors determined by the PCA, valid is false if fit fails
    
    '''
    return_data = np.empty((len(vtx_data),8))
    return_pca_data = np.empty((len(vtx_data),6))
    return_pca_nhits = np.empty((len(vtx_data),6))
    for idx, vtx in enumerate(vtx_data):
        filtered_hits_mask = norm(shower_hits[:,:3], vtx[:3]) < radius # select hits within a radius
        filtered_hits = shower_hits[filtered_hits_mask]
        if len(filtered_hits) == 0:
            return_data[idx] = np.array(list(vtx[:4]) + list([False,0,0,0]))
            return_pca_data[idx] = np.array(list(vtx[:4]) + list([False,0]))
            return_pca_nhits[idx] = np.array(list(vtx[:4]) + list([False,0]))
            continue
        associated_hits_mask = dbscan_find_primary(filtered_hits[:,:3], eps=eps, min_samples=min_samples) # associate hits with dbscan
        associated_hits = filtered_hits[associated_hits_mask]
        if len(associated_hits) <= 2:
            return_data[idx] = np.array(list(vtx[:4]) + list([False,0,0,0]))
            return_pca_data[idx] = np.array(list(vtx[:4]) + list([False,0]))
            return_pca_nhits[idx] = np.array(list(vtx[:4]) + list([False,0]))
            continue
        selected_hits = associated_hits
        pca_vecs, pca_vals = pca(associated_hits[:,:3])
        parity = compute_parity_flip(associated_hits[:,:3], pca_vecs[0], origin=vtx[:3]) # reverse vector in case it points in the antiparallel direction
        return_data[idx] = np.array(list(vtx[:4]) + [True] + list(pca_vecs[0]*parity))
        return_pca_data[idx] = np.array(list(vtx[:4]) + [True] + list([np.sqrt(pca_vals[1]**2 + pca_vals[2]**2)]))
        return_pca_nhits[idx] = np.array(list(vtx[:4]) + [True] + list([len(associated_hits)]))
    return return_data, return_pca_data, return_pca_nhits