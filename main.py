import numpy as np
import mnist
from network import Network

EPOCHS = 5
BATCH = 64
LR = 0.1


def accuracy(net, x, y):
    return (net.predict(x) == y).mean()


def main():
    try:
        x_train, y_train, x_test, y_test = mnist.load()
    except FileNotFoundError:
        print("MNIST not found in ./data — run ./get_mnist.sh first. Using random data so it still runs.")
        rng = np.random.default_rng(0)
        x_train, y_train = rng.random((1000, 784), dtype=np.float32), rng.integers(0, 10, 1000)
        x_test, y_test = x_train[:200], y_train[:200]

    print(f"train {x_train.shape}, test {x_test.shape}")
    net = Network([784, 128, 10])

    for epoch in range(EPOCHS):
        order = np.random.permutation(len(x_train))
        losses = []
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            losses.append(net.train_step(x_train[idx], y_train[idx], LR))
        print(f"epoch {epoch + 1}: loss {np.mean(losses):.4f}  test acc {accuracy(net, x_test, y_test):.3f}")

    zs, acts = net.forward_full(x_train[:64])
    print(len(zs), [z.shape for z in zs])
    print(len(acts), [a.shape for a in acts])
    print(np.allclose(net.forward(x_train[:64]), acts[-1]))
    print(net.cost(x_train[:64], y_train[:64]))
    net.back(x_train[:64],y_train[:64])

    




if __name__ == "__main__":
    main()
