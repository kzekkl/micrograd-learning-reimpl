# M5：训练 moons 分类器
import random
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from micrograd import nn
from micrograd.engine import Value

def make_moons(n=100, noise=0.1):
    l_t = [random.uniform(0, math.pi) for _ in range(n)]
    x_out = [math.cos(t) + random.gauss(0,noise) for t in l_t]
    y_out = [math.sin(t) + random.gauss(0,noise) for t in l_t]
    x_in = [1 - math.cos(t) + random.gauss(0,noise) for t in l_t]
    y_in = [0.5 - math.sin(t) + random.gauss(0,noise) for t in l_t]

    points = list(zip(x_out, y_out)) + list(zip(x_in, y_in))
    lable = [1]*n + [-1]*n
    pairs = list(zip(points, lable))
    random.shuffle(pairs)
    points_shuffled, lable_shuffled = zip(*pairs)
    points_shuffled = list(points_shuffled)
    lable_shuffled = list(lable_shuffled)
    return points_shuffled, lable_shuffled

def toValue(point):
    return Value(point[0]), Value(point[1])

def accuracy(model, X, y):
    right = 0
    for point, label in zip(X, y):
        score = model(list(toValue(point)))
        label_hat = 1 if score.data>0 else -1
        if label_hat == label:
            right += 1

    return right / len(X)


def plot_loss_curve(losses, path="loss_curve.png"):
    plt.figure(figsize=(7, 4))
    plt.plot(range(1, len(losses) + 1), losses, color="tab:blue")
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.title("Training loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def plot_decision_boundary(model, X_tr, y_tr, X_te, y_te,
                           path="decision_boundary.png", grid_size=50):
    xs = [p[0] for p in X_tr + X_te]
    ys = [p[1] for p in X_tr + X_te]
    pad = 0.3
    x_min, x_max = min(xs) - pad, max(xs) + pad
    y_min, y_max = min(ys) - pad, max(ys) + pad

    gx = np.linspace(x_min, x_max, grid_size)
    gy = np.linspace(y_min, y_max, grid_size)
    xx, yy = np.meshgrid(gx, gy)

    zz = np.zeros((grid_size, grid_size))
    for i, yv in enumerate(gy):
        for j, xv in enumerate(gx):
            score = model([Value(float(xv)), Value(float(yv))])
            zz[i, j] = score.data

    plt.figure(figsize=(7, 5.5))
    cs = plt.contourf(xx, yy, zz, levels=25, cmap="RdBu", alpha=0.45)
    plt.colorbar(cs, label="network score")
    plt.contour(xx, yy, zz, levels=[0], colors="k", linewidths=1.2)

    for label, color, name in [(1, "tab:blue", "class +1 (train)"),
                               (-1, "tab:orange", "class -1 (train)")]:
        pts = [p for p, l in zip(X_tr, y_tr) if l == label]
        plt.scatter([p[0] for p in pts], [p[1] for p in pts],
                    s=18, c=color, edgecolors="k", linewidths=0.4, label=name)

    for label, color, name in [(1, "tab:blue", "class +1 (test)"),
                               (-1, "tab:orange", "class -1 (test)")]:
        pts = [p for p, l in zip(X_te, y_te) if l == label]
        plt.scatter([p[0] for p in pts], [p[1] for p in pts],
                    s=42, c=color, marker="x", label=name)

    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    plt.xlabel("x1")
    plt.ylabel("x2")
    plt.title("Decision boundary (black line: score = 0)")
    plt.legend(loc="upper right", fontsize=8, ncol=2)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def main():
    random.seed(42)
    X, y = make_moons()
    split = int(0.8*len(X))
    X_tr = X[:split]
    y_tr = y[:split]
    X_te = X[split:]
    y_te = y[split:]  

    model = nn.MLP(2,[16,16,1])
    lr, alpha = 0.1, 1e-4
    losses = []
    for epoch in range(80):
        total = Value(0)

        for point, label in zip(X_tr, y_tr):
            score = model(list(toValue(point)))
            margin = label * score
            total += (1 - margin).relu()

        data_loss = total / len(X_tr)
        reg_loss = alpha * sum(p*p for p in model.parameters())
        loss = data_loss + reg_loss

        model.zero_grad()
        loss.backward()
        for p in model.parameters():
            p.data -= lr * p.grad

        losses.append(loss.data)
        if epoch % 10 == 0:
            print(epoch, loss.data)

    tr_acc = accuracy(model, X_tr, y_tr)
    te_acc = accuracy(model, X_te, y_te)
    print(f"train acc: {tr_acc:.2%} | test acc: {te_acc:.2%}")

    plot_loss_curve(losses, "loss_curve.png")
    plot_decision_boundary(model, X_tr, y_tr, X_te, y_te,
                           "decision_boundary.png", grid_size=50)
    print("saved: loss_curve.png, decision_boundary.png")

if __name__ == "__main__":
    main()
