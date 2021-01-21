from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
import numpy as np


def find_direction(start, points, max_dist):
    """
    Estimate the direction of fragment points using a small sphere around
    the staring points.

    Arguments
    ---------
    start: array_like, dim = (3, )
        voxel coordinates of starting poisition

    points: array_like, dim = (n, 3)
        a collection of voxels

    max_dist: float
        the size of sphere used to estimate the direction

    Returns
    -------
    v: ndarray, dim = (3, )
        estimated direction vector
        return (0, 0, 0) if there is not enought neigbor points

    selected: ndarray of bool, dim = (M, )
        mask array for direction calculation

    History
    -------
    DATE       WHO     WHAT
    2020.01.10 kvtsang Created
    2020.02.04 kvtsang Return mask array for selected voxels
    """
    dist = cdist([start], points)[0]

    selected = dist < max_dist
    neighbors = points[selected]

    # not enought voxels to estimate the directions
    if len(neighbors) < 5:
        return np.zeros(3), selected

    # the major eigenvector
    fit = PCA(n_components=1).fit(neighbors)
    v = fit.components_.squeeze(0)

    # determine the sign of vector from centorid minus start
    centroid = np.mean(points, axis=0)
    if v.dot(centroid - start) < 0:
        v *= -1

    return v, selected


def partition(A):
    """
    Given adjacent matrix A, partition a graph to its disconnected sub-graphs.

    Arguments
    ---------
    A: array_like, dim(A) = (n ,n)
        Adjacent matrix of a graph.

    Returns
    -------
    groups: list
        A list of list. Each sub-list is the node indices of a sub-graph.

    History
    -------
    DATE       WHO     WHAT
    2020.01.10 kvtsang Created


    Examples
    --------
    >>> A = [[0, 1, 1, 0, 0],
             [1, 0, 1, 0, 0],
             [1, 1, 0, 0, 0],
             [0, 0, 0, 0, 1],
             [0, 0, 0, 1, 0]]

    >>> partition(A)
    [[0, 1, 2], [3, 4]]
    """

    # TODO(2020-01-10 kvtsang) Check A = A.T
    A = np.asarray(A)
    n =  len(A)
    active = [True] * n
    idx = np.arange(n)
    groups = []

    # make boolean adjacent matrix
    B = (A != 0)

    # self connected
    np.fill_diagonal(B, True)

    while np.any(active):
        # first active row
        i = idx[active][0]

        # add curr row to a new group
        status = B[i].copy()
        group = [i]
        active[i] = False

        # find connected nodes
        for j in idx[active]:
            if np.any(status & B[j]):
                status |= B[j]
                group.append(j)
                active[j] = False

        groups.append(group)

    return groups


def create_tree(fragments, threshold_perp,  threshold_rad, veto_dist):
    """
    Create a tree-like structure for shower framgments.

    Assumption: each fragment (frag) is a list that contains
    - starting position (frag[1])
    - a collection of voxels (frag[2][:, :3]).

    Arguments
    ---------
    fragments: list
        list of fragments

    threshold_perp: float
        maximum perpendicular distance between voxels in a fragment and a direction vector

    threshold_rad: float
        maximumn radiation lenght between voxels in a fragment and a starting position
    
    veto_dist: float
        veto distance within starting point to avoid toching to the starting point
        The vetod voxels are used to calculate direction

    Returns
    -------
    parents: list, (n, )
        Indices to the parent fragments.

        parents[i] gives the index of closest fragment to i-th fragment,
        matched by backwoard projection from i-th starting point.

        parents[i] = -1 if there is no backward fragment found. In this case,
        the i-th fragment is the initial fragment of its cluster.


    History
    -------
    DATE       WHO     WHAT
    2020.01.10 kvtsang Created
    2020.01.13 kvtsang Use the closest projected distance for matching.
                       At most one backward fragment will be found.
    2020.02.24 kvtsang Skip back-to-back fragments matching.
    """
    n = len(fragments)
    parents = -np.ones(n, dtype=np.int)

    directions = []
    veto_masks = []
    for frag in fragments:
        v, mask = find_direction(frag[0], frag[1][:, :3], veto_dist)
        directions.append(v)
        veto_masks.append(~mask)

    for i, frag_i in enumerate(fragments):
        start = np.array(frag_i[0])
        voxels_i = frag_i[1][:, :3]

        # estimate the direction vector
        # skip for tiny fragment
        v = directions[i]
        if np.allclose(v, np.zeros(3)):
            continue

        last_score = 0.

        for j, frag_j in enumerate(fragments):
            if i == j:
                continue

            # skip back-to-back fragment pairs
            if v.dot(directions[j]) <= 0:
                continue

            veto_mask = veto_masks[j]
            if not np.any(veto_mask):
                continue

            voxels_j = frag_j[1][veto_mask, :3]
            dp = voxels_j - start

            # projection of frag_j's voxels to the direction of frag_i
            dist_proj = dp.dot(v)

            # only look at backward direction
            mask = dist_proj < 0
            if not np.any(mask):
                continue
            dist_proj = dist_proj[mask]

            # distances between frag_j and the starting of frag_i
            dist = cdist([start], voxels_j[mask]).squeeze(0)

            # cut on raditaion lenght
            mask2 = dist < threshold_rad
            if not np.any(mask2):
                continue
            dist = dist[mask2]
            dist_proj = dist_proj[mask2]

            # perpendicular distances between frag_j and the direction of frag_i
            dist2_perp = dist**2 - dist_proj**2

            # connect two fragment within a small perpendicular distance
            if dist2_perp.min() < threshold_perp**2:
                score = -dist_proj.min() # note: projected distance in backward
                if score >  last_score:
                    last_score = score
                    parents[i] = j
    return parents


def trace_tree(parents):
    """
    Separate non-connected part and trace a tree-like using parent map.

    Arguments
    ---------
    parents: list
        A relation array that maps to parent index.

    Returns
    -------
    output: dict
        key - the root of a tree
        values - list of indices belong to the same tree


    History
    -------
    DATE       WHO     WHAT
    2020.01.10 kvtsang Created
    2020.02.04 kvtsang Break 1-loop if i->j and j->i

    See also
    --------
    create_tree

    Notes
    -----
    This is not a tree implentation, e.g. no spliting and merging.
    It does not preserve any hierarchical structure except the root node.

    Examples
    --------
    >>> parents = [3, 0, -1, -1, 2, 5]
    >>> trace_tree(parents)
    3: [0, 3, 1], 2: [2, 4, 5]}


    This example shows two trees with roots of in node 2 and 3.

    3 - 0 - 1

        4
      /
    2
      \
        5
    """
    n = len(parents)
    fast_forward = [-1] * n
    output = {}

    # speical treatment if i->j and j->i
    for i in range(n):
        j = parents[i]
        if j != -1 and i == parents[j]:
            parents[i] = -1
            parents[j] = -1

    for i in range(n):
        curr = i
        head = None
        nodes = []

        cnt = 0
        while head is None:
            if cnt > n:
                raise RuntimeError('Fail to build tree. Check parents list to make sure there is not cycle.')

            jump = fast_forward[curr]
            # check if current node has visited with known head
            if jump != -1:
                head = jump
            else:
                nodes.append(curr)
                before = parents[curr]
                # check if current node is head
                if before == -1:
                    head = curr
                else:
                    curr = before

            cnt += 1

        # update cache for all visited nodes in this round
        for j in nodes:
            fast_forward[j] = head

        # combine nodes with the same head
        if head in output:
            output[head].extend(nodes)
        else:
            output[head] = nodes
    return output


def group_fragments(fragments, dist_prep=15, dist_rad=150, veto_dist=10):
    """
    Merge shower fragments using backward direction matching.

    Arguments
    ---------
    fragments: list
        list of shower fragments

    dist_prep: float
        threshold on perpendicular distance, default = 15 voxels

    dist_rad: float
        threshold on radiation length, default = 150 voxels

    veto_dist: float
        veto distance within starting point to avoid toching to the starting point
        The vetod voxels are used to calculate direction
        default = 10 voxels

    Returns
    -------
    roots: list
        List of indices of root node (i.e. starting of a merged fragment group)

    groups: list
        List of list, where each sub-list is a list of fragment indices
        that are merged to the some group.

    pairs: ndarray (n ,2)
        List of pair (i, j), where j-th fragment is the closest from
        backward matching i-th fragment.

    See also
    --------
    create_tree, trace_tree

    History
    -------
    DATE       WHO     WHAT
    2020.01.10 kvtsang Created
    2020.01.13 kvtsang Modified using tree-like structure
                       Returned root of each merged fragments
    """
    parents = create_tree(fragments, dist_prep, dist_rad, veto_dist)
    psuedo_trees = trace_tree(parents)

    roots = list(psuedo_trees.keys())
    groups = list(psuedo_trees.values())

    # do not return root nodes as a pair
    mask = parents != -1
    pairs = np.column_stack((np.where(mask)[0], parents[mask]))
    return roots, groups, pairs
