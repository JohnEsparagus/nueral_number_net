import gzip
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")


def _open(name):
    path = os.path.join(DATA_DIR, name)
    if os.path.exists(path + ".gz"):
        return gzip.open(path + ".gz", "rb")
    return open(path, "rb")


def _images(name):
    with _open(name) as f:
        data = np.frombuffer(f.read(), np.uint8, offset=16)
    return data.reshape(-1, 784).astype(np.float32) / 255.0  


def _labels(name):
    with _open(name) as f:
        return np.frombuffer(f.read(), np.uint8, offset=8).astype(np.int64)  


def load():
    #x_train, y_train, x_test, y_test
    return (
        _images("train-images-idx3-ubyte"),
        _labels("train-labels-idx1-ubyte"),
        _images("t10k-images-idx3-ubyte"),
        _labels("t10k-labels-idx1-ubyte"),
    )
