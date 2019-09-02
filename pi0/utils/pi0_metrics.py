import numpy as np


def normalize_vector(vec):
    """
    Given n-dimensional vector, return normalized unit vector. 

    Inputs:
        - vec (np.array): N x d array, where N is the number of vectors. 

    Returns:
        - normed_vec: N x d normalized vector array.
    """
    v = np.atleast_2d(vec)
    v = v / np.linalg.norm(v, axis=1).reshape(v.shape[0], 1)
    return v


def cosine_similarity(vec1, vec2):
    """
    NOTE: Input vectors need not be normalized to unity. 

    Inputs:
        - vec1: (N, 3) np.array, where each row contains a
        predicted/truth direction vector. 
        - vec2: (N, 3) np.array, where each row contains a 
        truth/predicted direction vector. 

    Returns:
        - similarity: (N, ) np.array, where each row is the
        cosine similarity metric. 
    """
    assert vec1.shape == vec2.shape
    # Normalize to unit vectors
    vec1 = vec1 / np.linalg.norm(vec1, axis=1).reshape(vec1.shape[0], 1)
    vec2 = vec2 / np.linalg.norm(vec2, axis=1).reshape(vec2.shape[0], 1)

    return np.diag(np.dot(vec1, vec2.T))

def angular_similarity(vec1, vec2, weight=1.0):

    csim = cosine_similarity(vec1, vec2)
    return 1 - np.arccos(csim) / (np.pi / weight)


def f_score(purity, efficiency):
    """
    Computes the F1-score for clustering accuracy, defined as the
    harmonic mean of purity and efficiency. 

    Inputs:
        - purity (float): purity score for given event
        - efficiency (float): efficiency score for given event

    Returns:
        - fscore: self-expanatory.
    """
    return 2 * (purity * efficiency) / (purity + efficiency)