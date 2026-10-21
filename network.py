import numpy as np


class Network:

    def __init__(self, sizes, seed=0):
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 0.01, (n_out, n_in)) for n_in, n_out in zip(sizes, sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in sizes[1:]]

    def forward(self, x):
        return np.zeros((x.shape[0], self.sizes[-1]))

    def train_step(self, x, y, lr):
        return 0.0

    def predict(self, x):
        return self.forward(x).argmax(axis=1)
