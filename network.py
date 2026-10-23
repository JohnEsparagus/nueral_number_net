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
        zs, acts = self.forward_full(x)
        return acts[-1]
    
    def forward_full(self,x):
        acts = [x]
        zs = []
        a = x
        last = len(self.W) - 1 #twice

        for l, (w, b) in enumerate(zip(self.W, self.b)):
            z = a @ w.T + b     
            zs.append(z)   
            if (l == last):
                a = z 
            else:
                a = np.maximum(0,z)
            acts.append(a)
        return zs, acts


        
    
    def cost(self, x, y):
        #given x and y, we find 
        z = self.forward(x)
        p = self.softmax(z)
        #cross entryo
        
        
        p = np.clip(p,1e-12,1)

        #if 0 we are cooked so we clamp
        correct_set = p[np.arange(len(y)),y]
        loss = -np.log(correct_set)

        return np.mean(loss)

    def back(self, x, y, lr):
        zs, acts = self.forward_full(x)
        #logits are in zs
        p = self.softmax(zs[-1])
        one_hot = np.eye(10)[y]

        # we know logits so calc parameters.

        n = x.shape[0]
        l = len(self.sizes)
        delta = (p - one_hot) / n

        db_W = [None] * len(self.W)
        db_b = [None] * len(self.b)

        for i in reversed(range(len(self.W))):
            db_W[i] = (delta.T @ acts[i])
            db_b[i] = np.sum(delta, axis=0)

            if i > 0:
                delta = (delta @ self.W[i]) * (zs[0] > 0)

        return db_W, db_b


        


    def train_step(self, x, y, lr):
        loss = self.cost(x,y)
        db_W, db_b = self.back(x,y,lr)
        for w, dw in zip(self.W, db_W):
            w -= lr * dw
        for b, db in zip(self.b, db_b):
            b -= lr * db
        
        return loss
        


    def predict(self, x):
        return self.forward(x).argmax(axis=1)
