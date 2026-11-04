import numpy as np


def soft_IPQ(alpha: np.ndarray) -> float:
    """
    Soft-IPQ implementation.
    
    Args:
        alpha: (H, W, C) or (H, W) array of alpha values.
    Returns:
        float: Soft-IPQ score.
    """
    assert alpha.ndim in (2, 3), "alpha must be 2D or 3D (H, W[, C])"
    alpha = alpha.astype(float)
    if alpha.ndim == 2:
        alpha = alpha[:, :, None]

    # calculate magnitude and gradient magnitude
    magnitudes = np.sqrt(np.sum(np.square(alpha), axis=-1))
    gy, gx = np.gradient(alpha, axis=(0, 1))
    grad_magnitudes = np.sqrt(np.sum(gx ** 2 + gy ** 2, axis=-1))

    # Accumulate magnitudes
    A = np.sum(magnitudes)
    L = np.sum(grad_magnitudes)
    if A == 0:
        return 0.0
    else:
        return (L ** 2) / (4 * np.pi * A)