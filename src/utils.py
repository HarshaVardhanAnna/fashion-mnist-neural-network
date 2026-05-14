import numpy as np
from keras.datasets import fashion_mnist

CLASS_NAMES = [
    'T-shirt/top', 'Trouser',  'Pullover', 'Dress',     'Coat',
    'Sandal',      'Shirt',    'Sneaker',  'Bag',        'Ankle boot'
]

def load_data(val_split: float = 0.1):
    (X_tr, y_tr), (X_test, y_test) = fashion_mnist.load_data()

    # Pixel normalisation
    X_tr   = X_tr.astype(np.float64)   / 255.0
    X_test = X_test.astype(np.float64) / 255.0

    # Flatten + transpose
    X_tr   = X_tr.reshape(-1, 784).T
    X_test = X_test.reshape(-1, 784).T

    # Shuffle before splitting so val set is representative
    n_total = X_tr.shape[1]
    rng = np.random.default_rng(42)
    idx = rng.permutation(n_total)
    X_tr, y_tr = X_tr[:, idx], y_tr[idx]

    n_val = int(n_total * val_split)
    X_val,   y_val   = X_tr[:, :n_val],   y_tr[:n_val]
    X_train, y_train = X_tr[:, n_val:],   y_tr[n_val:]

    return X_train, y_train, X_val, y_val, X_test, y_test


def get_batches(X, y, batch_size: int, shuffle: bool = True):
    """
    Generator that yields (X_batch, y_batch) mini-batches.
    """
    n = X.shape[1]
    idx = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        end       = min(start + batch_size, n)
        batch_idx = idx[start:end]
        yield X[:, batch_idx], y[batch_idx]