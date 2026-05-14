"""
Six gradient-based optimizers for the neural network.

All optimizers share a common interface:
    optimizer.update(weights, biases, grads)
    weights : list of W matrices
    biases  : list of b vectors
    grads   : list of (dW, db) tuples from nn.backward()
"""
import numpy as np


class SGD:
    """
    Vanilla Stochastic Gradient Descent.
    Simplest possible update; no memory of past gradients.
    """
    name = "sgd"

    def __init__(self, lr: float = 1e-3):
        self.lr = lr

    def update(self, weights, biases, grads):
        for i, (dW, db) in enumerate(grads):
            weights[i] -= self.lr * dW
            biases[i]  -= self.lr * db


class Momentum:
    """
    SGD with Polyak Momentum.

    The velocity v accumulates an exponential moving average of past
    gradients — this dampens oscillations and accelerates convergence in
    consistent gradient directions.
    """
    name = "momentum"

    def __init__(self, lr: float = 1e-3, beta: float = 0.9):
        self.lr   = lr
        self.beta = beta
        self.v_W  = self.v_b = None

    def _init(self, weights, biases):
        self.v_W = [np.zeros_like(W) for W in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, grads):
        if self.v_W is None:
            self._init(weights, biases)
        for i, (dW, db) in enumerate(grads):
            self.v_W[i] = self.beta * self.v_W[i] + dW
            self.v_b[i] = self.beta * self.v_b[i] + db
            weights[i] -= self.lr * self.v_W[i]
            biases[i]  -= self.lr * self.v_b[i]


class Nesterov:
    """
    Nesterov Accelerated Gradient Descent.
    """
    name        = "nesterov"
    is_nesterov = True                  # flag used by train.py

    def __init__(self, lr: float = 1e-3, beta: float = 0.9):
        self.lr   = lr
        self.beta = beta
        self.v_W  = self.v_b = None

    def _init(self, weights, biases):
        self.v_W = [np.zeros_like(W) for W in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

    def apply_lookahead(self, weights, biases):
        """Shift weights to lookahead position: W ← W - β·v"""
        if self.v_W is None:
            self._init(weights, biases)
        for i in range(len(weights)):
            weights[i] -= self.beta * self.v_W[i]
            biases[i]  -= self.beta * self.v_b[i]

    def update(self, weights, biases, grads):
        """Undo the lookahead shift, then apply the NAG velocity update."""
        for i, (dW, db) in enumerate(grads):
            # Restore: undo the lookahead shift
            weights[i] += self.beta * self.v_W[i]
            biases[i]  += self.beta * self.v_b[i]
            # Update velocity
            self.v_W[i] = self.beta * self.v_W[i] + dW
            self.v_b[i] = self.beta * self.v_b[i] + db
            # Final update
            weights[i] -= self.lr * self.v_W[i]
            biases[i]  -= self.lr * self.v_b[i]


class RMSProp:
    """
    RMSProp.
   
    Normalises the learning rate per parameter by its recent gradient
    magnitude. Useful when different parameters have very different scales.
    """
    name = "rmsprop"

    def __init__(self, lr: float = 1e-3, rho: float = 0.9, eps: float = 1e-8):
        self.lr  = lr
        self.rho = rho
        self.eps = eps
        self.s_W = self.s_b = None

    def _init(self, weights, biases):
        self.s_W = [np.zeros_like(W) for W in weights]
        self.s_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, grads):
        if self.s_W is None:
            self._init(weights, biases)
        for i, (dW, db) in enumerate(grads):
            self.s_W[i] = self.rho * self.s_W[i] + (1 - self.rho) * dW ** 2
            self.s_b[i] = self.rho * self.s_b[i] + (1 - self.rho) * db ** 2
            weights[i] -= self.lr * dW / (np.sqrt(self.s_W[i]) + self.eps)
            biases[i]  -= self.lr * db / (np.sqrt(self.s_b[i]) + self.eps)


class Adam:
    """
    Adam — Adaptive Moment Estimation.

    Maintains both a first moment (mean) and second moment (variance) of
    gradients, with bias correction for the initial zero-initialised state.
    """
    name = "adam"

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m_W   = self.m_b = None
        self.v_W   = self.v_b = None
        self.t     = 0

    def _init(self, weights, biases):
        self.m_W = [np.zeros_like(W) for W in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_W = [np.zeros_like(W) for W in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, grads):
        if self.m_W is None:
            self._init(weights, biases)
        self.t += 1
        # Precompute bias-corrected learning rate (numerically convenient)
        lr_t = (self.lr
                * np.sqrt(1 - self.beta2 ** self.t)
                / (1 - self.beta1 ** self.t))

        for i, (dW, db) in enumerate(grads):
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * dW ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * db ** 2
            weights[i] -= lr_t * self.m_W[i] / (np.sqrt(self.v_W[i]) + self.eps)
            biases[i]  -= lr_t * self.m_b[i] / (np.sqrt(self.v_b[i]) + self.eps)


class Nadam:
    """
    Nadam — Nesterov + Adam
    This gives Adam the same "future gradient" correction that NAG
    provides for vanilla momentum.
    """
    name = "nadam"

    def __init__(self, lr: float = 1e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.lr    = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps   = eps
        self.m_W   = self.m_b = None
        self.v_W   = self.v_b = None
        self.t     = 0

    def _init(self, weights, biases):
        self.m_W = [np.zeros_like(W) for W in weights]
        self.m_b = [np.zeros_like(b) for b in biases]
        self.v_W = [np.zeros_like(W) for W in weights]
        self.v_b = [np.zeros_like(b) for b in biases]

    def update(self, weights, biases, grads):
        if self.m_W is None:
            self._init(weights, biases)
        self.t += 1

        for i, (dW, db) in enumerate(grads):
            # Update moments
            self.m_W[i] = self.beta1 * self.m_W[i] + (1 - self.beta1) * dW
            self.m_b[i] = self.beta1 * self.m_b[i] + (1 - self.beta1) * db
            self.v_W[i] = self.beta2 * self.v_W[i] + (1 - self.beta2) * dW ** 2
            self.v_b[i] = self.beta2 * self.v_b[i] + (1 - self.beta2) * db ** 2

            # Nesterov-corrected first moment (looks one step ahead)
            m_hat_W = (self.beta1 * self.m_W[i] / (1 - self.beta1 ** (self.t + 1))
                       + (1 - self.beta1) * dW  / (1 - self.beta1 ** self.t))
            m_hat_b = (self.beta1 * self.m_b[i] / (1 - self.beta1 ** (self.t + 1))
                       + (1 - self.beta1) * db  / (1 - self.beta1 ** self.t))

            # Bias-corrected second moment
            v_hat_W = self.v_W[i] / (1 - self.beta2 ** self.t)
            v_hat_b = self.v_b[i] / (1 - self.beta2 ** self.t)

            weights[i] -= self.lr * m_hat_W / (np.sqrt(v_hat_W) + self.eps)
            biases[i]  -= self.lr * m_hat_b / (np.sqrt(v_hat_b) + self.eps)


# Registry
_OPTIMIZERS = {
    "sgd":      SGD,
    "momentum": Momentum,
    "nesterov": Nesterov,
    "rmsprop":  RMSProp,
    "adam":     Adam,
    "nadam":    Nadam,
}

def get_optimizer(name: str, **kwargs):
    """Instantiate an optimizer by name. Extra kwargs are forwarded."""
    if name not in _OPTIMIZERS:
        raise ValueError(f"Unknown optimizer '{name}'. "
                         f"Choose from: {list(_OPTIMIZERS)}")
    return _OPTIMIZERS[name](**kwargs)