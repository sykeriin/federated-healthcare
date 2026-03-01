"""
compression.py
SVD-based low-rank approximation for model update compression.
This is the core technical contribution — compresses gradient/weight
updates by 70-90% before transmission over low-bandwidth links.
"""

import time
import numpy as np
from typing import List, Tuple, Dict
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Core SVD compression
# ─────────────────────────────────────────────────────────────────────────────

CompressedTensor = Tuple[np.ndarray, np.ndarray, np.ndarray, tuple]


def svd_compress(tensor: np.ndarray, rank_ratio: float = 0.1) -> CompressedTensor:
    """
    Compress a weight/gradient tensor using low-rank SVD approximation.

    Args:
        tensor:     The numpy array to compress (any shape).
        rank_ratio: Fraction of singular values to keep (0.0 - 1.0).
                    0.1 = keep top 10% → ~80-90% size reduction.

    Returns:
        Tuple of (U_r, S_r, Vh_r, original_shape) for reconstruction.
    """
    original_shape = tensor.shape

    # Flatten to 2D matrix for SVD
    matrix = tensor.reshape(tensor.shape[0], -1) if tensor.ndim > 1 else tensor.reshape(1, -1)
    m, n = matrix.shape

    # Determine rank to keep
    max_rank = min(m, n)
    rank = max(1, int(max_rank * rank_ratio))

    # SVD decomposition — uses AMD ROCm if torch.linalg available on GPU
    try:
        t = torch.tensor(matrix, dtype=torch.float32)
        # Use GPU if available (AMD ROCm or CUDA)
        if torch.cuda.is_available():
            t = t.cuda()
        U, S, Vh = torch.linalg.svd(t, full_matrices=False)
        U = U.cpu().numpy()
        S = S.cpu().numpy()
        Vh = Vh.cpu().numpy()
    except Exception:
        # CPU fallback with numpy
        U, S, Vh = np.linalg.svd(matrix, full_matrices=False)

    # Keep only top-r components
    U_r  = U[:, :rank]
    S_r  = S[:rank]
    Vh_r = Vh[:rank, :]

    return (U_r, S_r, Vh_r, original_shape)


def svd_decompress(compressed: CompressedTensor) -> np.ndarray:
    """
    Reconstruct the original tensor from its compressed SVD components.

    Args:
        compressed: Tuple of (U_r, S_r, Vh_r, original_shape)

    Returns:
        Reconstructed numpy array in the original shape.
    """
    U_r, S_r, Vh_r, original_shape = compressed
    matrix = U_r @ np.diag(S_r) @ Vh_r
    return matrix.reshape(original_shape)


# ─────────────────────────────────────────────────────────────────────────────
# Compress / decompress a full model update (list of tensors)
# ─────────────────────────────────────────────────────────────────────────────

def compress_model_update(
    parameters: List[np.ndarray],
    rank_ratio: float = 0.1,
    min_size: int = 100,
) -> Tuple[List, Dict]:
    """
    Compress a full set of model parameters (one per layer).

    Layers smaller than `min_size` elements are sent uncompressed
    (compression overhead isn't worth it for tiny bias vectors).

    Returns:
        compressed_params: List of either CompressedTensor or plain np.ndarray
        stats:             Dict with compression ratio and timing info
    """
    start = time.time()
    compressed_params = []
    original_bytes = 0
    compressed_bytes = 0

    for param in parameters:
        original_bytes += param.nbytes

        if param.size >= min_size and param.ndim >= 2:
            comp = svd_compress(param, rank_ratio=rank_ratio)
            U_r, S_r, Vh_r, _ = comp
            compressed_bytes += U_r.nbytes + S_r.nbytes + Vh_r.nbytes
            compressed_params.append(("svd", comp))
        else:
            # Small layers (biases etc.) — send as-is
            compressed_bytes += param.nbytes
            compressed_params.append(("raw", param))

    elapsed = time.time() - start
    ratio = 1.0 - (compressed_bytes / max(original_bytes, 1))

    stats = {
        "original_kb":   original_bytes / 1024,
        "compressed_kb": compressed_bytes / 1024,
        "ratio":         ratio,
        "reduction_pct": ratio * 100,
        "time_ms":       elapsed * 1000,
    }

    return compressed_params, stats


def decompress_model_update(compressed_params: List) -> List[np.ndarray]:
    """
    Reconstruct model parameters from a compressed update.
    """
    result = []
    for tag, data in compressed_params:
        if tag == "svd":
            result.append(svd_decompress(data))
        else:
            result.append(data)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Differential Privacy noise injection
# ─────────────────────────────────────────────────────────────────────────────

def add_dp_noise(
    parameters: List[np.ndarray],
    noise_multiplier: float = 0.5,
    max_grad_norm: float = 1.0,
) -> List[np.ndarray]:
    """
    Add calibrated Gaussian noise for Differential Privacy.

    This is the lightweight manual version — for full DP accounting
    (epsilon-delta guarantees) use Opacus in client.py instead.

    Args:
        parameters:      List of numpy arrays (model update).
        noise_multiplier: Scale of noise relative to max_grad_norm.
        max_grad_norm:   Gradient clipping threshold (L2).

    Returns:
        Noised parameter list.
    """
    noised = []
    for param in parameters:
        # Clip gradient norm
        norm = np.linalg.norm(param)
        if norm > max_grad_norm:
            param = param * (max_grad_norm / norm)

        # Add Gaussian noise
        noise = np.random.normal(
            loc=0.0,
            scale=noise_multiplier * max_grad_norm,
            size=param.shape,
        ).astype(param.dtype)
        noised.append(param + noise)

    return noised


# ─────────────────────────────────────────────────────────────────────────────
# Hardware detection for adaptive compression
# ─────────────────────────────────────────────────────────────────────────────

def detect_hardware() -> Dict:
    """
    Detect available compute hardware and return recommended settings.
    This is the hardware-aware scheduling piece.
    """
    info = {
        "has_gpu":       False,
        "gpu_name":      None,
        "rank_ratio":    0.1,   # aggressive compression for CPU
        "local_epochs":  1,     # fewer epochs on weak hardware
        "device":        "cpu",
    }

    if torch.cuda.is_available():
        info["has_gpu"]      = True
        info["gpu_name"]     = torch.cuda.get_device_name(0)
        info["rank_ratio"]   = 0.2   # less compression needed — more bandwidth can be used
        info["local_epochs"] = 3     # more epochs on powerful hardware
        info["device"]       = "cuda"

    return info


def print_compression_stats(stats: Dict, clinic_id: int) -> None:
    """Pretty-print compression stats for demo output."""
    print(f"\n[Clinic {clinic_id}] 📉 Compression Stats:")
    print(f"  Original size:   {stats['original_kb']:.1f} KB")
    print(f"  Compressed size: {stats['compressed_kb']:.1f} KB")
    print(f"  Reduction:       {stats['reduction_pct']:.1f}%")
    print(f"  Time taken:      {stats['time_ms']:.1f} ms")
