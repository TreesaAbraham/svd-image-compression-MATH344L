import numpy as np

def rank_k_approx(A, k):
    """Return the rank-k SVD approximation of A."""
    U, s, Vt = np.linalg.svd(A, full_matrices=False)
    return U[:, :k] @ np.diag(s[:k]) @ Vt[:k, :]

def storage(m, n, k):
    """Number of values needed to store a rank-k approximation."""
    return k * (m + n + 1)

def compression_ratio(m, n, k):
    """Original number of values divided by compressed number of values."""
    return (m * n) / storage(m, n, k)

def rel_error(A, B):
    """Relative Frobenius error when B approximates A."""
    return np.linalg.norm(A - B) / np.linalg.norm(A)

def predicted_error(s, k):
    """Relative rank-k error computed from the singular values."""
    return np.sqrt(np.sum(s[k:]**2) / np.sum(s**2))

def k_for_error(s, target):
  "Smallest k whose relative error is at most target"
  for k in range(1, len(s)+1):
    if predicted_error(s, k) <= target:
      return k
