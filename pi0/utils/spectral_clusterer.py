import numpy as np
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
import itertools
from scipy.sparse import diags, coo_matrix
from scipy.spatial import distance_matrix
from sklearn.cluster import DBSCAN
from scipy.spatial import Delaunay

def cartesian_product(*arrays):
    la = len(arrays)
    dtype = np.result_type(*arrays)
    arr = np.empty([len(a) for a in arrays] + [la], dtype=dtype)
    for i, a in enumerate(np.ix_(*arrays)):
        arr[...,i] = a
    return arr.reshape(-1, la)

def dist_metric(v1, v2, characteristic_length=45):
    norms = np.linalg.norm(v2-v1, axis=1)
    weights = np.exp(-norms/characteristic_length)
    return weights

def direction_metric(edges, positions, dists, neighbor_threshold=0.95, eps_regularization=0.01):
    n = len(positions)
    neighbors = np.where(dists > neighbor_threshold)[0]
    neighbor_pairs = np.array([edges[:, 0][neighbors], edges[:, 1][neighbors]]).T
    F = np.zeros((n, 9))
    for i in range(n):
        v_filter = np.where(neighbor_pairs == i)
        neighbor_edges = neighbor_pairs[v_filter[0]]
        flip_filter = np.where(v_filter[1] == 1)
        neighbor_edges[flip_filter] = np.flip(neighbor_edges[flip_filter], axis=1)
        vector = positions[neighbor_edges[:, 1]] - positions[neighbor_edges[:, 0]]
        Fv = np.sum(np.einsum('ij,ik->ijk',vector,vector), axis=0)
        if len(neighbor_edges) > 0:
            Fv = Fv / len(neighbor_edges)
        Fv = (1 - eps_regularization)*Fv + eps_regularization*np.identity(3)
        F[i] = Fv.flatten()
    
    weights = np.einsum("ij,ij->i", F[edges[:, 0]], F[edges[:, 1]])
#     print(np.amin(weights))
#     print(np.amax(weights))
#     print('number of weights', len(weights))
    return weights

def adjacency(positions, em_indices, edges=None, direction_weight=False, em_weight=100.0, dbscan_weight=0.02, dbscan_eps=10):
    n = len(positions)
    nodes = np.arange(n)
    
    simplices = Delaunay(positions).simplices
    simplices.sort()
    edges = set()
    for s in simplices:
        edges |= set(itertools.combinations(s, 2))
    edges = np.array(list(edges))
#     print('# of edges', len(edges))
    
#     edges = np.vstack(np.triu_indices(n, k=1)).T

    dists = dist_metric(positions[edges[:, 0]], positions[edges[:, 1]])
    weights = dists
#     print('distance weights', weights)
    if direction_weight:
        directions = direction_metric(edges, positions, dists)
        weights = directions*dists
#     print('directions', directions)
    
    # create adjacency matrix where weights are nonzero
#     on_edges = np.where(weights != 0)
#     edges = edges[on_edges]
#     print('Fraction of edges turned on:', len(edges)/len(weights))
#     weights = weights[on_edges]
    
    # this is only upper triangular
    A_upper_half = coo_matrix((weights, (edges[:, 0], edges[:, 1])), shape=(n, n), dtype=weights.dtype).tocsc()
    A_lower_half = coo_matrix((weights, (edges[:, 1], edges[:, 0])), shape=(n, n), dtype=weights.dtype).tocsc()
    A = A_upper_half + A_lower_half
    
    
    # add DBSCAN bias
    # closely attach EM primaries to their DBSCAN cluster, repulse different EM primary clusters
    clusters = DBSCAN(eps=dbscan_eps, min_samples=2).fit(positions).labels_
    primary_clusters = clusters[em_indices]
    dbscan_indices = []
    dbscan_weights = []
    for i in range(np.amax(clusters)):
        same_cluster = np.zeros(n)
        dbscan_filter = np.where(clusters == i)[0]
        if i in primary_clusters:
            # sparsely repulse all other primary clusters
            for j in primary_clusters:
                if j != i:
                    db_filter = np.where(clusters == j)[0]
                    sparse_indices = cartesian_product(db_filter, dbscan_filter)
#                     print('sparse_indices', len(sparse_indices))
                    dbscan_indices.extend(sparse_indices)
                    dbscan_weights.extend(len(sparse_indices) * [-1 * em_weight])
        indices = list(itertools.combinations(dbscan_filter, 2))
        dbscan_indices.extend(indices)
        dbscan_weights.extend(len(indices) * [dbscan_weight])
    dbscan_indices = np.array(dbscan_indices)
    dbscan_weights = np.array(dbscan_weights)
    if len(dbscan_weights) > 0:
        A += coo_matrix((dbscan_weights, (dbscan_indices[:, 0], dbscan_indices[:, 1])), shape=(n, n), dtype=weights.dtype).tocsc()
        A += coo_matrix((dbscan_weights, (dbscan_indices[:, 1], dbscan_indices[:, 0])), shape=(n, n), dtype=weights.dtype).tocsc()
#     print('DBSCAN processed')
    
    
#     indices = np.array(list(itertools.combinations(em_indices, 2)))
#     if len(indices) > 0:
#         D_vec = np.array(A.sum(axis=1)).flatten()
#         em_repulsion = -10.0 * np.ones(len(indices))
#         A += coo_matrix((em_repulsion, (indices[:, 0], indices[:, 1])), shape=(n, n), dtype=weights.dtype).tocsc()
#         A += coo_matrix((em_repulsion, (indices[:, 1], indices[:, 0])), shape=(n, n), dtype=weights.dtype).tocsc()
        
    
    D_vec = np.abs(np.array(A.sum(axis=1)).flatten())
    D_norm_vec = 1/np.sqrt(D_vec)
    D = diags(D_vec)
    D_norm = diags(D_norm_vec)
    
    A_norm = D_norm * A * D_norm
#     print('A normalized')
    
    return A_norm

def get_eigenvectors(A, dims):
    try:
        vals, vecs = eigsh(A, k=dims, which='LA')
    except:
        return None
#     print(vals.tolist())
#     print(vecs)
#     vecs = vecs[:, :-1]  # remove eigenvector corresponding to zero eigenvalue
    return vecs

def transform_space(positions, em_primary_positions, dims, max_em_primary_distance_to_voxel=3, em_weight=100.0, dbscan_weight=0.02, dbscan_eps=10):
    # find closest voxels to EM primaries
    distances = distance_matrix(em_primary_positions, positions)
    min_indices = np.argmin(distances, axis=1)
    min_vals = np.amin(distances, axis=1)
    em_indices = min_indices[np.where(min_vals < max_em_primary_distance_to_voxel)]
#     print(em_indices)
    
    A = adjacency(positions, em_indices, em_weight=em_weight, dbscan_weight=dbscan_weight, dbscan_eps=dbscan_eps)
#     print('made adjacency')
    vecs = get_eigenvectors(A, dims)
    
    return vecs



from scipy.linalg import qr
from sklearn.neighbors import NearestNeighbors

def cluster(positions, em_primary_positions, params=[25.985069618575174, -2.0, 1.0, 1.0]):
    """
    positions: Nx3 array of EM shower voxel positions
    em_primaries: Nx3 array of EM primary positions
    
    returns a tuple (arr of length len(em_primaries), arr of length len(positions)) corresponding to EM primary labels and the voxel labels; note that each voxel has a unique label
    """
    lv = transform_space(positions, em_primary_positions, 15, em_weight=10.0**params[1], dbscan_weight=10.0**params[2], dbscan_eps=10.0**params[3])
    if lv is None:
        return None
    for i in range(len(lv[0]) - 1):
        lv[:, i] -= np.mean(lv[:, i])
        lv[:, i] /= np.std(lv[:, i])
    # print(lv)

    vecs = lv[:, (-len(em_primary_positions)-1):-1]

    Q, R, P = qr(vecs.T, pivoting=True)

    cluster_assignments = np.zeros(len(P))
    for i in range(len(P)):
        p = P[i]
        col = R[:, i]
        col_ind = np.argmax(np.abs(col))
        cluster_assignments[p] = np.sign(col[col_ind])*(col_ind + 1) # add 1 since -0 == 0

    # merge spectral clusters with directly neighboring voxels
    coords = positions
    neigh = NearestNeighbors(n_neighbors=6, radius=1.0)
    neigh.fit(coords)
    in_radius = neigh.radius_neighbors(coords)[1]

    labels_to_merge = []
    candidate_mergers = []
    for point in in_radius:
        sp_labels = np.unique(cluster_assignments[point])
        if len(sp_labels) > 1:
            merge = sp_labels.tolist()
            candidate_mergers.extend(list(itertools.combinations(merge, 2)))
    candidate_mergers = np.array(candidate_mergers)
    if len(candidate_mergers) > 0:
        pairs, counts = np.unique(candidate_mergers, axis=0, return_counts=True)
        pairs = pairs[np.where(counts > params[0])]
        for merge in pairs:
            merge_index = -1
            for i in range(len(labels_to_merge)):
                for m in merge:
                    if m in labels_to_merge[i]:
                        merge_index = i
                        break
                if merge_index > -1:
                    break
            if merge_index == -1:
                labels_to_merge.append(merge)
            else:
                labels_to_merge[merge_index] = list(set().union(labels_to_merge[merge_index], merge))

        for merge in labels_to_merge:
            for i in range(1, len(merge)):
                cluster_assignments[np.where(cluster_assignments == merge[i])] = merge[0]

    return cluster_assignments