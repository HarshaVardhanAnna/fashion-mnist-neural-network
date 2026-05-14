"""
Flexible feedforward neural network with forward and backward passes.
"""
import numpy as np
from src.activations import get_activation, softmax


class NeuralNetwork:
    def __init__(
        self,
        input_size:   int,
        hidden_sizes: list,
        output_size:  int  = 10,
        activation:   str  = "relu",
        weight_init:  str  = "xavier",
    ):
        self.layer_sizes   = [input_size] + list(hidden_sizes) + [output_size]
        self.n_layers      = len(self.layer_sizes) - 1   # number of weight matrices
        self.activation_fn, self.activation_deriv = get_activation(activation)
        self.weight_init   = weight_init

        self.weights: list[np.ndarray] = []   # W[l]: shape (out_l, in_l)
        self.biases:  list[np.ndarray] = []   # b[l]: shape (out_l, 1)
        self._init_params()

    # Weight Initialisation
    def _init_params(self):
        """
        Xavier init: std = sqrt(2 / (fan_in + fan_out))
        Keeps variance stable across layers — critical for sigmoid/tanh.
        Random init: small Gaussian, std = 0.01 (baseline comparison).
        """
        for i in range(self.n_layers):
            fan_in  = self.layer_sizes[i]
            fan_out = self.layer_sizes[i + 1]

            if self.weight_init == "xavier":
                std = np.sqrt(2.0 / (fan_in + fan_out))
            else:                               # 'random'
                std = 0.01

            self.weights.append(np.random.randn(fan_out, fan_in) * std)
            self.biases.append(np.zeros((fan_out, 1)))

    # Forward Pass
    def forward(self, X: np.ndarray):
        """
        Compute the forward pass and cache intermediate values for backprop.

        Parameters
        ----------
        X : (784, batch_size)

        Returns
        -------
        y_pred : (10, batch_size)   softmax probabilities
        cache  : dict
            'h' — list of post-activation outputs, h[0]=X, h[l]=activation(a[l])
            'a' — list of pre-activation values,  a[l] = W[l] @ h[l-1] + b[l]
        """
        cache: dict = {"h": [X], "a": []}
        h = X

        # Hidden layers  (all but the last weight matrix)
        for i in range(self.n_layers - 1):
            a = self.weights[i] @ h + self.biases[i]   # pre-activation
            h = self.activation_fn(a)                   # post-activation
            cache["a"].append(a)
            cache["h"].append(h)

        # Output layer: linear → softmax (no activation stored for 'a')
        a_out  = self.weights[-1] @ h + self.biases[-1]
        y_pred = softmax(a_out)
        cache["a"].append(a_out)   # needed if using squared-error loss
        cache["h"].append(y_pred)

        return y_pred, cache

    # Backward Pass
    def backward(
        self,
        grad_output: np.ndarray,
        cache:       dict,
        weight_decay: float = 0.0,
    ) -> list:
        """
        Backpropagation through the network.

        Parameters
        ----------
        grad_output  : (10, batch_size)
            dL/da_L — gradient of loss w.r.t. the *pre-softmax* logits.
            This is precomputed by the loss function's gradient helper
            (which folds in the softmax Jacobian analytically).

        weight_decay : λ for L2 regularisation.
            Adds λ·W to dW, which is equivalent to a weight-decay penalty.

        Returns
        -------
        grads : list of (dW, db) tuples in *forward layer order* (0 … L-1)
        """
        grads_rev = []
        delta     = grad_output                 # δ starts at output layer

        for i in range(self.n_layers - 1, -1, -1):
            h_prev = cache["h"][i]              # input to layer i: (fan_in, batch)
            dW     = delta @ h_prev.T + weight_decay * self.weights[i]
            db     = np.sum(delta, axis=1, keepdims=True)
            grads_rev.append((dW, db))

            if i > 0:   # propagate δ to the layer below
                dh    = self.weights[i].T @ delta
                delta = dh * self.activation_deriv(cache["a"][i - 1])

        return list(reversed(grads_rev))        # forward order: layer 0 … L-1

    # Inference
    def predict(self, X: np.ndarray) -> np.ndarray:
        """X: (784, n) → predicted class indices (n,)"""
        y_pred, _ = self.forward(X)
        return np.argmax(y_pred, axis=0)

    def accuracy(self, X: np.ndarray, y: np.ndarray) -> float:
        return float(np.mean(self.predict(X) == y))