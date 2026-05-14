"""
All activation functions and their element-wise derivatives.
Each activation operates on numpy arrays of any shape.
"""
import numpy as np


# Sigmoid
def sigmoid(z: np.ndarray) -> np.ndarray:
    """Numerically stable: avoids exp overflow for large negative z."""
    return np.where(
        z >= 0,
        1.0 / (1.0 + np.exp(-z)),
        np.exp(z) / (1.0 + np.exp(z))
    )

def sigmoid_derivative(z: np.ndarray) -> np.ndarray:
    s = sigmoid(z)
    return s * (1.0 - s)


# Tanh
def tanh_fn(z: np.ndarray) -> np.ndarray:
    return np.tanh(z)

def tanh_derivative(z: np.ndarray) -> np.ndarray:
    return 1.0 - np.tanh(z) ** 2


# ReLU
def relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, z)

def relu_derivative(z: np.ndarray) -> np.ndarray:
    return (z > 0).astype(np.float64)


# Softmax (output layer only)
def softmax(z: np.ndarray) -> np.ndarray:
    """
    Numerically stable column-wise softmax.
    z : (n_classes, batch_size)
    """
    z_shifted = z - np.max(z, axis=0, keepdims=True)   # subtract max per sample
    exp_z     = np.exp(z_shifted)
    return exp_z / np.sum(exp_z, axis=0, keepdims=True)


# Registry
_ACTIVATIONS = {
    "sigmoid": (sigmoid,  sigmoid_derivative),
    "tanh":    (tanh_fn,  tanh_derivative),
    "relu":    (relu,     relu_derivative),
}

def get_activation(name: str):
    """Returns (forward_fn, derivative_fn) for the named activation."""
    if name not in _ACTIVATIONS:
        raise ValueError(f"Unknown activation '{name}'. "
                         f"Choose from: {list(_ACTIVATIONS)}")
    return _ACTIVATIONS[name]