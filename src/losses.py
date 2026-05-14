"""
Loss functions and their gradients w.r.t. the pre-softmax logits (a_L).
"""
import numpy as np


def _one_hot(y: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer labels → one-hot matrix: (n_classes, batch_size)."""
    oh = np.zeros((n_classes, len(y)))
    oh[y, np.arange(len(y))] = 1.0
    return oh


# Cross-Entropy
def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray,
                       eps: float = 1e-15) -> float:
    """
    Standard multi-class cross-entropy.
    y_pred: softmax probabilities (10, batch)
    y_true: integer labels        (batch,)
    """
    n    = len(y_true)
    y_p  = np.clip(y_pred, eps, 1.0 - eps)
    oh   = _one_hot(y_true, y_pred.shape[0])
    return -np.sum(oh * np.log(y_p)) / n


def cross_entropy_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    n  = len(y_true)
    oh = _one_hot(y_true, y_pred.shape[0])
    return (y_pred - oh) / n


# Squared Error
def squared_error_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    """Mean squared error between softmax output and one-hot targets."""
    n  = len(y_true)
    oh = _one_hot(y_true, y_pred.shape[0])
    return np.sum((y_pred - oh) ** 2) / (2.0 * n)


def squared_error_grad(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    """
    This avoids a costly per-sample loop; all operations are batched.
    Shape: (10, batch_size)
    """
    n      = len(y_true)
    oh     = _one_hot(y_true, y_pred.shape[0])
    dL_dy  = (y_pred - oh) / n                              # (10, batch)
    dot    = np.sum(y_pred * dL_dy, axis=0, keepdims=True)  # (1,  batch)
    return y_pred * (dL_dy - dot)                            # (10, batch)


# Registry 
_LOSSES = {
    "cross_entropy": (cross_entropy_loss, cross_entropy_grad),
    "squared_error": (squared_error_loss, squared_error_grad),
}

def get_loss(name: str):
    if name not in _LOSSES:
        raise ValueError(f"Unknown loss '{name}'. Choose from: {list(_LOSSES)}")
    return _LOSSES[name]