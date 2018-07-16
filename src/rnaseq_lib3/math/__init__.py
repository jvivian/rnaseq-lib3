import numpy as np


def l2norm(x: float, pad: float = 1.0) -> float:
    """Log2 normalization function"""
    return np.log2(x + pad)
