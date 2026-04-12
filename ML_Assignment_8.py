import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPClassifier


def step(y):
    return 1 if y > 0 else 0

def bipolar_step(y):
    return 1 if y > 0 else (-1 if y < 0 else 0)

def sigmoid(y):
    return 1 / (1 + np.exp(-y))

def relu(y):
    return max(0, y)

def tanh_fn(y):
    return np.tanh(y)

def leaky_relu(y):
    return y if y > 0 else 0.01 * y

def activate(y, name):
    fns = {'step': step, 'bipolar': bipolar_step, 'sigmoid': sigmoid,
           'relu': relu, 'tanh': tanh_fn, 'leaky_relu': leaky_relu}
    return fns[name](y)

# Weighted sum (bias + inputs)
def summation(inputs, weights):
    return weights[0] + np.dot(inputs, weights[1:])

# Error = target - output
def error(target, output):
    return target - output

# Sum of squared errors
def sse(targets, outputs):
    return sum((t - o) ** 2 for t, o in zip(targets, outputs))



X = np.array([[0,0],[0,1],[1,0],[1,1]])
y_and = [0, 0, 0, 1]
y_xor = [0, 1, 1, 0]


def train_perceptron(X, y, act='step', lr=0.05, max_epochs=1000, tol=0.002):
    w = np.array([10.0, 0.2, -0.75])   # w0=bias, w1, w2
    errors = []

    for epoch in range(max_epochs):
        outputs = []
        for xi, ti in zip(X, y):
            net = summation(xi, w)
            out = activate(net, act)
            err = error(ti, out)
            w[0] += lr * err          # update bias
            w[1] += lr * err * xi[0]  # update w1
            w[2] += lr * err * xi[1]  # update w2
            outputs.append(out)

        total_error = sse(y, outputs)
        errors.append(total_error)

        if total_error <= tol:
            print(f"  [{act}] Converged at epoch {epoch+1}, Error={total_error:.4f}")
            return w, errors, epoch+1

    print(f"  [{act}] Did NOT converge in {max_epochs} epochs")
    return w, errors, max_epochs



print("=== A2: AND Gate - Step Activation ===")
w, errs, conv = train_perceptron(X, y_and, act='step')
print(f"  Final Weights: {w}")

plt.figure()
plt.plot(errs)
plt.title("A2: AND Gate - Step Activation")
plt.xlabel("Epoch"); plt.ylabel("SSE"); plt.grid(True)
plt.savefig("a2_and_step.png"); plt.show()



print("\n=== A3: AND Gate - Activation Comparison ===")
plt.figure()
for act in ['step', 'bipolar', 'sigmoid', 'relu']:
    _, errs, _ = train_perceptron(X, y_and, act=act)
    plt.plot(errs, label=act)

plt.title("A3: Activation Comparison"); plt.xlabel("Epoch"); plt.ylabel("SSE")
plt.legend(); plt.grid(True)
plt.savefig("a3_activations.png"); plt.show()



print("\n=== A4: Varying Learning Rate ===")
lrs = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
conv_epochs = []

for lr in lrs:
    _, _, conv = train_perceptron(X, y_and, act='step', lr=lr)
    conv_epochs.append(conv)

plt.figure()
plt.plot(lrs, conv_epochs, marker='o')
plt.title("A4: Learning Rate vs Epochs to Converge")
plt.xlabel("Learning Rate"); plt.ylabel("Epochs"); plt.grid(True)
plt.savefig("a4_lr.png"); plt.show()



print("\n=== A5: XOR Gate - Activation Comparison ===")
for act in ['step', 'bipolar', 'sigmoid', 'relu']:
    train_perceptron(X, y_xor, act=act)



print("\n=== A6: Customer Classification ===")

X_cust = np.array([
    [20, 6, 2, 386], [16, 3, 6, 289], [27, 6, 2, 393],
    [19, 1, 2, 110], [24, 4, 2, 280], [22, 1, 5, 167],
    [15, 4, 2, 271], [18, 4, 2, 274], [21, 1, 4, 148],
    [16, 2, 4, 198]
], dtype=float)
y_cust = [1, 1, 1, 0, 1, 0, 1, 1, 0, 0]

# Normalize
X_cust = (X_cust - X_cust.mean(axis=0)) / X_cust.std(axis=0)

def train_perceptron_n(X, y, lr=0.1, act='sigmoid', max_epochs=1000, tol=0.002):
    """Perceptron for any number of features"""
    w = np.zeros(X.shape[1] + 1)   # bias + n weights
    errors = []

    for epoch in range(max_epochs):
        outputs = []
        for xi, ti in zip(X, y):
            net = w[0] + np.dot(xi, w[1:])
            out = activate(net, act)
            err = error(ti, out)
            w[0] += lr * err
            w[1:] += lr * err * xi
            outputs.append(out)

        total_error = sse(y, outputs)
        errors.append(total_error)
        if total_error <= tol:
            print(f"  Converged at epoch {epoch+1}")
            return w, errors

    print(f"  Did not converge in {max_epochs} epochs")
    return w, errors

w_cust, _ = train_perceptron_n(X_cust, y_cust)
preds = [1 if sigmoid(w_cust[0] + np.dot(xi, w_cust[1:])) >= 0.5 else 0 for xi in X_cust]
print(f"  Predictions: {preds}")
print(f"  Actual:      {y_cust}")
print(f"  Accuracy:    {sum(p==t for p,t in zip(preds,y_cust))/len(y_cust)*100:.1f}%")



print("\n=== A7: Pseudo-Inverse ===")
X_b = np.hstack([np.ones((X_cust.shape[0], 1)), X_cust])  # add bias column
w_pinv = np.linalg.pinv(X_b) @ np.array(y_cust, dtype=float)
preds_pinv = [1 if (w_pinv[0] + np.dot(xi, w_pinv[1:])) >= 0.5 else 0 for xi in X_cust]
print(f"  Predictions: {preds_pinv}")
print(f"  Accuracy:    {sum(p==t for p,t in zip(preds_pinv,y_cust))/len(y_cust)*100:.1f}%")



print("\n=== A8: Backpropagation - AND Gate ===")

def train_backprop(X, y, lr=0.05, max_epochs=1000, tol=0.002):
    """2-input → 2-hidden → 1-output with sigmoid"""
    np.random.seed(42)
    V = np.random.uniform(-0.05, 0.05, (3, 2))  # input->hidden weights
    W = np.random.uniform(-0.05, 0.05, (3, 1))  # hidden->output weights
    errors = []

    for epoch in range(max_epochs):
        total_err = 0
        for xi, ti in zip(X, y):
            # --- Forward ---
            h = sigmoid(V[0] + xi @ V[1:])          # hidden outputs
            o = sigmoid(W[0,0] + h @ W[1:,0])       # final output

            total_err += (ti - o) ** 2

            # --- Backward ---
            d_o = o * (1 - o) * (ti - o)            # output delta
            d_h = h * (1 - h) * (W[1:,0] * d_o)    # hidden deltas

            # --- Update weights ---
            W[0,0] += lr * d_o
            W[1:,0] += lr * d_o * h
            V[0] += lr * d_h
            V[1:] += lr * np.outer(xi, d_h)

        errors.append(total_err)
        if total_err <= tol:
            print(f"  Converged at epoch {epoch+1}")
            return V, W, errors

    print(f"  Did not converge in {max_epochs} epochs")
    return V, W, errors

V, W, bp_errs = train_backprop(X, y_and)

plt.figure()
plt.plot(bp_errs, color='green')
plt.title("A8: Backpropagation - AND Gate")
plt.xlabel("Epoch"); plt.ylabel("SSE"); plt.grid(True)
plt.savefig("a8_backprop.png"); plt.show()



print("\n=== A9: Backpropagation - XOR Gate ===")
train_backprop(X, y_xor)



print("\n=== A10: 2-Output Backprop - AND Gate ===")
# 0 → [1,0],  1 → [0,1]
y_2out = [[1,0],[1,0],[1,0],[0,1]]

def train_backprop_2out(X, y, lr=0.05, max_epochs=1000, tol=0.002):
    """2-input → 2-hidden → 2-output backprop"""
    np.random.seed(42)
    V = np.random.uniform(-0.05, 0.05, (3, 2))
    W = np.random.uniform(-0.05, 0.05, (3, 2))

    for epoch in range(max_epochs):
        total_err = 0
        for xi, ti in zip(X, y):
            ti = np.array(ti, dtype=float)
            h = sigmoid(V[0] + xi @ V[1:])
            o = sigmoid(W[0] + h @ W[1:])

            total_err += np.sum((ti - o) ** 2)

            d_o = o * (1 - o) * (ti - o)
            d_h = h * (1 - h) * (W[1:] @ d_o)

            W[0] += lr * d_o
            W[1:] += lr * np.outer(h, d_o)
            V[0] += lr * d_h
            V[1:] += lr * np.outer(xi, d_h)

        if total_err <= tol:
            print(f"  Converged at epoch {epoch+1}")
            return

    print(f"  Did not converge in {max_epochs} epochs")

train_backprop_2out(X, y_2out)



print("\n=== A11: MLPClassifier ===")

mlp_and = MLPClassifier(hidden_layer_sizes=(4,), max_iter=1000, random_state=42)
mlp_and.fit(X, y_and)
print(f"  AND Gate -> Predicted: {mlp_and.predict(X).tolist()}  Actual: {y_and}")

mlp_xor = MLPClassifier(hidden_layer_sizes=(4,), max_iter=5000, random_state=42)
mlp_xor.fit(X, y_xor)
print(f"  XOR Gate -> Predicted: {mlp_xor.predict(X).tolist()}  Actual: {y_xor}")



print("\n=== A12: MLPClassifier - Customer Data ===")
mlp_cust = MLPClassifier(hidden_layer_sizes=(8, 4), max_iter=2000, random_state=42)
mlp_cust.fit(X_cust, y_cust)
pred = mlp_cust.predict(X_cust).tolist()
acc  = sum(p==t for p,t in zip(pred,y_cust)) / len(y_cust) * 100
print(f"  Predicted: {pred}")
print(f"  Actual:    {y_cust}")
print(f"  Accuracy:  {acc:.1f}%")

print("\nDone! All plots saved.")