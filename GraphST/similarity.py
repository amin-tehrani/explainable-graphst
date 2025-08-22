import torch
from typing import Optional

EPS = 1e-12

def _as_vec(x: torch.Tensor) -> torch.Tensor:
    """Ensure 1D vector (D,) from possible shapes like (1, D) or (D, 1)."""
    x = x.squeeze()
    if x.dim() != 1:
        raise ValueError(f"Expected a 1D vector, got shape {tuple(x.shape)}")
    return x

def l2_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = _as_vec(x), _as_vec(y)
    return torch.norm(x - y, p=2)

def l1_distance(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    x, y = _as_vec(x), _as_vec(y)
    return torch.norm(x - y, p=1)

def cosine_distance(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    x, y = _as_vec(x), _as_vec(y)
    x_norm = x / (x.norm(p=2) + eps)
    y_norm = y / (y.norm(p=2) + eps)
    cos_sim = (x_norm * y_norm).sum()
    return 1.0 - cos_sim  # in [0, 2]

def angular_distance(x: torch.Tensor, y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """Angle between vectors in radians, in [0, π]."""
    x, y = _as_vec(x), _as_vec(y)
    x_norm = x / (x.norm(p=2) + eps)
    y_norm = y / (y.norm(p=2) + eps)
    cos_sim = (x_norm * y_norm).sum().clamp(-1.0, 1.0)
    return torch.acos(cos_sim)

def mahalanobis_distance(
    x: torch.Tensor,
    y: torch.Tensor,
    inv_cov: Optional[torch.Tensor] = None,
    eps: float = 1e-6,
) -> torch.Tensor:
    """
    d_M(x,y) = sqrt( (x-y)^T * Sigma^{-1} * (x-y) )
    - inv_cov: (D,D) inverse covariance or any SPD matrix^{-1}.
      If None, uses identity (reduces to L2).
    """
    x, y = _as_vec(x), _as_vec(y)
    diff = (x - y).unsqueeze(0)  # (1, D)
    if inv_cov is None:
        return torch.norm(diff, p=2)
    # Ensure symmetric positive-definite numerically
    inv_cov = 0.5 * (inv_cov + inv_cov.T)
    # Quadratic form
    qf = diff @ inv_cov @ diff.T  # (1,1)
    # Clamp to avoid tiny negative due to numerical error
    return torch.sqrt(qf.clamp_min(0.0)).squeeze()

class BilinearMetric(torch.nn.Module):
    """
    Learnable Mahalanobis-like metric: d_W(x,y) = sqrt( (x-y)^T W (x-y) ), W = A^T A (SPD).
    """
    def __init__(self, D: int):
        super().__init__()
        self.A = torch.nn.Parameter(torch.eye(D))  # initialize as identity

    def forward(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        x, y = _as_vec(x), _as_vec(y)
        diff = (x - y)
        # W = A^T A is SPD; compute sqrt(diff^T W diff)
        Ad = self.A @ diff
        return torch.norm(Ad, p=2)


def pairwise_cdist(X: torch.Tensor, Y: torch.Tensor, p: float = 2.0) -> torch.Tensor:
    """
    Pairwise p-norm distances between two batches.
    X: (N, D), Y: (M, D) -> (N, M)
    """
    return torch.cdist(X, Y, p=p)

def pairwise_cosine_distance(X: torch.Tensor, Y: torch.Tensor, eps: float = EPS) -> torch.Tensor:
    """
    Pairwise cosine distance: 1 - cosine_similarity for two batches.
    X: (N, D), Y: (M, D) -> (N, M)
    """
    Xn = X / (X.norm(dim=1, keepdim=True) + eps)
    Yn = Y / (Y.norm(dim=1, keepdim=True) + eps)
    cos = Xn @ Yn.T  # (N, M)
    return 1.0 - cos.clamp(-1.0, 1.0)
