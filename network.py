import numpy as np


class Network:

    def __init__(self, sizes, seed=0):
        self.sizes = sizes
        rng = np.random.default_rng(seed)
        self.W = [rng.normal(0, 0.01, (n_out, n_in)) for n_in, n_out in zip(sizes, sizes[1:])]
        self.b = [np.zeros(n_out) for n_out in sizes[1:]]

    def print_ascii(self,img,x,y):
        arr = [[" " for j in range(x)] for i in range(y)]
        img = img.reshape(x,y)
        for i in range(x):
            for j in range(y):
                p = img[i][j]
                if p > 0.5:
                    arr[i][j] = "#"

        for row in arr:
            print("".join(row))

        return None
    

                
    def weighted_sum(self, image, layer=0):
        #784 pixels
        #128*784 weights
        # 128 biases
        # a is input image so... its just the pixels
        a = image.reshape(-1)
        z = self.W[layer] @ a + self.b[layer]
        print(z)



    def forward(self, x):

        return np.zeros((x.shape[0], self.sizes[-1]))

    def train_step(self, x, y, lr):
        return 0.0

    def predict(self, x):
        return self.forward(x).argmax(axis=1)
