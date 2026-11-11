"""pytorch3d.ops stand-in: only knn_points / knn_gather, implemented with torch.cdist."""

import torch


def knn_points(p1, p2, K=1, return_nn=False, return_sorted=True, **kwargs):
    """Neighbours of each point in p2 among p1.

    p1: (N, P1, 3) points to search, p2: (N, P2, 3) queries.
    Returns squared distances (N, P2, K), indices (N, P2, K) and optionally (N, P2, K, 3).
    """
    dists2 = (p2.unsqueeze(2) - p1.unsqueeze(1)).pow(2).sum(-1)  # (N, P2, P1)
    dists2, idx = torch.topk(dists2, K, dim=-1, largest=False, sorted=return_sorted)
    nn = knn_gather(p1, idx) if return_nn else None
    return dists2, idx, nn


def knn_gather(points, idx):
    """Gather (N, P2, K) indices from (N, P1, 3) -> (N, P2, K, 3)."""
    return points.unsqueeze(1).expand(-1, idx.shape[1], -1, -1).gather(
        2, idx.unsqueeze(-1).expand(-1, -1, -1, points.shape[-1]))
