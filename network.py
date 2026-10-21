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
        return z


    def softmax(self, z):
        #for the total digits, what sthe probilbity its that

        shifted =z - z.max(axis =-1, keepdims=True)
        exp_z = np.exp(shifted)

        return exp_z / np.sum(exp_z, axis=-1, keepdims=True)
            

    def forward(self, x):
        #batch,784
        #every row is an image
        a = x
        last = len(self.W) - 1 #twice
        for l, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w.T + b        
            if (l == last):
                a = z 
            else:
                a = np.maximum(0,z)
        return a
    
    
    def cost(self, x):
        return 0.0

    def train_step(self, x, y, lr):
        return 0.0


    def predict(self, x):
        return self.forward(x).argmax(axis=1)
