import torch
from typing import Optional

EPS = 1e-12

def _as_vec(x: torch.Tensor) -> torch.Tensor:
    x = x.squeeze()
    if x.dim() != 1:
        raise ValueError(f"Expected a 1D vector, got shape {tuple(x.shape)}")
    return x

# ---------- Single-pair similarities (return scalar tensors) ----------

def cosine_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x, y = _as_vec(x), _as_vec(y)
    xn = x / (x.norm(p=2) + eps)
    yn = y / (y.norm(p=2) + eps)
    return (xn * yn).sum().clamp(-1.0, 1.0)  # in [-1, 1]

def dot_similarity(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = _as_vec(x), _as_vec(y)
    return (x * y).sum()  # unbounded; scale-sensitive

def angular_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Map angle to [0,1]: 1 means identical direction, 0 means opposite."""
    c = cosine_similarity(x, y, eps=eps)          # [-1, 1]
    angle = torch.acos(c.clamp(-1.0, 1.0))        # [0, π]
    return 1.0 - angle / torch.pi                 # [0, 1]

def rbf_similarity(x: torch.Tensor, y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    """
    exp(-gamma * ||x - y||^2).
    gamma = 1/(2*sigma^2). Larger gamma -> tighter neighborhood.
    """
    x, y = _as_vec(x), _as_vec(y)
    diff2 = torch.sum((x - y) ** 2)
    return torch.exp(-gamma * diff2)

def mahalanobis_rbf_similarity(
    x: torch.Tensor,
    y: torch.Tensor,
    inv_cov: torch.Tensor,
    gamma: float = 1.0,
) -> torch.Tensor:
    """
    exp(-gamma * (x - y)^T Σ^{-1} (x - y)).
    inv_cov: (D,D) positive-definite inverse covariance (or metric).
    """
    x, y = _as_vec(x), _as_vec(y)
    diff = (x - y).unsqueeze(0)           # (1, D)
    inv_cov = 0.5 * (inv_cov + inv_cov.T) # symmetrize
    qf = (diff @ inv_cov @ diff.T).squeeze()  # scalar
    qf = torch.clamp(qf, min=0.0)
    return torch.exp(-gamma * qf)

def pearson_similarity(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Pearson correlation in [-1,1]; shift to [0,1] if needed via (r+1)/2.
    """
    x, y = _as_vec(x), _as_vec(y)
    x = (x - x.mean()) / (x.std(unbiased=False) + eps)
    y = (y - y.mean()) / (y.std(unbiased=False) + eps)
    return (x * y).mean().clamp(-1.0, 1.0)

# ---------- Pairwise (batched) similarities ----------
# X: (N, D), Y: (M, D) -> (N, M)

def pairwise_cosine_similarity(X: torch.Tensor, Y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    Xn = X / (X.norm(dim=1, keepdim=True) + eps)
    Yn = Y / (Y.norm(dim=1, keepdim=True) + eps)
    return (Xn @ Yn.T).clamp(-1.0, 1.0)  # (N, M)

def pairwise_dot_similarity(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    return X @ Y.T

def pairwise_rbf_similarity(X: torch.Tensor, Y: torch.Tensor, gamma: float = 1.0) -> torch.Tensor:
    # ||x - y||^2 = ||x||^2 + ||y||^2 - 2 x·y
    X2 = (X * X).sum(dim=1, keepdim=True)       # (N,1)
    Y2 = (Y * Y).sum(dim=1, keepdim=True).T     # (1,M)
    XY = X @ Y.T                                # (N,M)
    dist2 = X2 + Y2 - 2 * XY
    dist2 = torch.clamp(dist2, min=0.0)
    return torch.exp(-gamma * dist2)

def pairwise_mahalanobis_rbf_similarity(
    X: torch.Tensor, Y: torch.Tensor, inv_cov: torch.Tensor, gamma: float = 1.0
) -> torch.Tensor:
    """
    Compute exp(-gamma * (x - y)^T Σ^{-1} (x - y)) for all pairs.
    Uses (A x - A y)^2 trick with A = chol(Σ^{-1}) for speed.
    """
    # Factor inv_cov as A^T A (Cholesky handles SPD)
    A = torch.linalg.cholesky(0.5 * (inv_cov + inv_cov.T))
    Xp = X @ A.T   # (N, D)
    Yp = Y @ A.T   # (M, D)
    return pairwise_rbf_similarity(Xp, Yp, gamma=gamma)
