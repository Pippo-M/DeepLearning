import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
np.random.seed(4)

X, y_binary = make_blobs(n_samples=300, centers=2, cluster_std=1.2)
y = 2 * y_binary - 1

w = np.zeros(X.shape[1])
b = 0.0
lr = 0.01
C = 1.0
epochs = 500
losses = []

for _ in range(epochs):
    margins = y * (X.dot(w) + b)
    hinge = np.maximum(0, 1 - margins)
    loss = 0.5 * np.dot(w, w) + C * hinge.mean()
    losses.append(loss)
    mask = margins < 1
    if mask.any():
        dw = w - (C / len(X)) * X[mask].T.dot(y[mask])
        db = -(C / len(X)) * y[mask].sum()
    else:
        dw = w
        db = 0.0
    w -= lr * dw
    b -= lr * db

plt.figure()
plt.plot(losses, color="green")
plt.title("Linear SVM (hinge) — Objective")
plt.xlabel("Epoch")
plt.ylabel("Objective")
plt.grid(True)
plt.show()

xx, yy = np.meshgrid(
    np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 200),
    np.linspace(X[:, 1].min() - 1, X[:, 1].max() + 1, 200)
)
grid = np.c_[xx.ravel(), yy.ravel()]
scores = grid.dot(w) + b

plt.figure()
plt.contourf(xx, yy, (scores.reshape(xx.shape) > 0).astype(int), cmap="Greens")
plt.scatter(X[:, 0], X[:, 1], c=y_binary, cmap="Greens",
            edgecolor="black", alpha=0.85)
plt.title("Linear SVM — Decision Boundary (approx)")
plt.xlabel("x1")
plt.ylabel("x2")
plt.show()