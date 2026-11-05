import json
import os
import numpy as np
import mnist
from network import Network

EPOCHS = 10
BATCH = 64
LR = 0.1


def main():
    x_train, y_train, x_test, y_test = mnist.load()
    net = Network([784, 128, 10])

    for epoch in range(EPOCHS):
        order = np.random.permutation(len(x_train))
        for i in range(0, len(order), BATCH):
            idx = order[i:i + BATCH]
            net.train_step(x_train[idx], y_train[idx], LR)
        acc = (net.predict(x_test) == y_test).mean()
        print(f"epoch {epoch + 1}: test acc {acc:.3f}")

    # the browser demo reads these as plain nested lists
    weights = {
        "W": [np.round(w, 5).tolist() for w in net.W],
        "b": [np.round(b, 5).tolist() for b in net.b],
        "test_acc": round(float(acc), 4),
    }
    out = os.path.join(os.path.dirname(__file__), "docs", "weights.js")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write("const WEIGHTS = " + json.dumps(weights) + ";\n")
    print(f"saved {out}")


if __name__ == "__main__":
    main()
