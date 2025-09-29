import torch
import numpy as np
from typing import Union

__all__ = [
    '_row_softmax',
    'prep_fixed_adj',
    '_cheb_supports_from_S',
    '_make_ring_adj',
    '_laplacian_eigvecs',
    '_to_torch_float',
]



def _row_softmax(mat: torch.Tensor) -> torch.Tensor:
    """
    Applies row-wise softmax to a matrix.

    Args:
        mat (torch.Tensor): The input tensor.

    Returns:
        torch.Tensor: The tensor with softmax applied to its last dimension.
    """
    return torch.softmax(mat, dim=-1)

def prep_fixed_adj(A: Union[np.ndarray, torch.Tensor], device) -> torch.Tensor:
    """
    Prepares a fixed adjacency matrix for use in the model.
    Converts a NumPy array or Tensor to a non-negative float tensor and applies
    row-wise softmax for normalization.

    Args:
        A (Union[np.ndarray, torch.Tensor]): The adjacency matrix.
        device: The target torch device.

    Returns:
        torch.Tensor: The processed and normalized adjacency matrix.
    """
    if isinstance(A, np.ndarray):
        A = torch.from_numpy(A.astype(np.float32))
    A = torch.clamp(A.to(device=device, dtype=torch.float32), 0.0)
    return _row_softmax(A)

def _cheb_supports_from_S(S: torch.Tensor, K: int) -> torch.Tensor:
    """
    Computes the Chebyshev polynomial basis of a graph for Chebyshev GCNs.
    The basis is computed up to order K-1 using the recurrence relation:
    T_k(S) = 2 * S @ T_{k-1}(S) - T_{k-2}(S), with T_0(S) = I and T_1(S) = S.

    Args:
        S (torch.Tensor): The scaled and shifted Laplacian matrix (e.g., 2*L/lmax - I).
        K (int): The order of the Chebyshev polynomial.

    Returns:
        torch.Tensor: A tensor of shape [K, N, N] containing the Chebyshev supports.
    """
    N = S.size(0)
    if K == 1:
        return torch.eye(N, device=S.device, dtype=S.dtype).unsqueeze(0)
    sup = [torch.eye(N, device=S.device, dtype=S.dtype), S]
    for _ in range(2, K):
        sup.append(2 * S @ sup[-1] - sup[-2])
    return torch.stack(sup)

def _make_ring_adj(n: int, device) -> torch.Tensor:
    """
    Creates an adjacency matrix for a ring graph with n nodes.
    In a ring graph, each node is connected to its two immediate neighbors.

    Args:
        n (int): The number of nodes in the graph.
        device: The target torch device.

    Returns:
        torch.Tensor: The adjacency matrix of the ring graph.
    """
    idx = torch.arange(n, device=device)
    A = torch.zeros((n, n), device=device)
    A[idx, (idx - 1) % n] = 1.0
    A[idx, (idx + 1) % n] = 1.0
    return A

def _laplacian_eigvecs(A: torch.Tensor, k: int) -> torch.Tensor:
    """
    Computes the first k eigenvectors of the normalized graph Laplacian.
    These eigenvectors form the basis for the Graph Fourier Transform.

    The normalized Laplacian is computed as L = I - D^(-1/2) * A * D^(-1/2).

    Args:
        A (torch.Tensor): The adjacency matrix.
        k (int): The number of eigenvectors to compute.

    Returns:
        torch.Tensor: A tensor of shape [N, k] containing the first k eigenvectors.
    """
    A = torch.clamp(0.5 * (A + A.T), 0.0)
    deg = torch.clamp(A.sum(-1), min=1e-6)
    Dm12 = torch.diag(torch.pow(deg, -0.5))
    L = torch.eye(A.size(0), device=A.device) - Dm12 @ A @ Dm12
    _, evecs = torch.linalg.eigh(L)
    return evecs[:, :min(k, A.size(0))].contiguous()

def _to_torch_float(a, device):
    """
    A utility function to convert an input to a torch.float32 tensor
    on a specified device. Handles None inputs gracefully.

    Args:
        a: The input to convert (can be Tensor, NumPy array, list, etc.).
        device: The target torch device.

    Returns:
        torch.Tensor or None: The converted float tensor, or None if input is None.
    """
    if a is None: return None
    if torch.is_tensor(a): return a.to(device=device, dtype=torch.float32)
    return torch.tensor(a, dtype=torch.float32, device=device)