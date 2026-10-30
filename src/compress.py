import numpy as np

def rank_k_approx(A, k):
    """Return the rank-k SVD approximation of A."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]
