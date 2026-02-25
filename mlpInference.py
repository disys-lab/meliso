import time, os
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

# ---------- numerically-stable softmax ----------
def softmax(x):
    z = x - np.max(x)
    e = np.exp(z)
    return e / np.sum(e)

# ---------- keep next-layer inputs in [0,1] ----------
def to_unit_interval(v, eps=1e-12):
    v = np.asarray(v, dtype=np.float64)
    v = np.clip(v, 0.0, None)       # ReLU outputs are nonnegative; clamp just in case
    vmax = np.max(v)
    if vmax < eps:
        return np.zeros_like(v)
    return v / vmax

# ---------- one memristor MVM that returns SCALED output ----------
def mem_mvm_scaled(x_scaled, W_scaled, y_path=None):
    mv = MatVecSolver(xvec=x_scaled, mat=W_scaled)
    mv.initializeVec()
    mv.initializeMat()
    # IMPORTANT: stay in scaled domain (no inverse/unscale)
    mv.matVec(correction=False)
    if y_path is None:
        base = os.environ.get("TMPDIR", ".")
        y_path = os.path.join(base, "y_mem_result.txt")
        y_scaled = np.loadtxt(y_path)   # robust: default whitespace delimiter
        if y_scaled.ndim > 1:
            y_scaled = y_scaled[:, 0]
        mv.finalize()
    return y_scaled

# ---------- one linear layer run in scaled domain ----------
def run_layer_scaled(x_in_scaled, W_scaled, b, alpha=1.0, beta=1.0,relu=True):
    """
    """
    y_scaled = mem_mvm_scaled(x_in_scaled, W_scaled)  # scaled-domain MVM result
    y = alpha * y_scaled + b + beta                   # add bias in real domain
    if relu:
        y = np.maximum(y, 0.0)
    x_next = to_unit_interval(y)                      # keep next layer input in [0,1]
    return y, x_next

# ---------- paths ----------
run_id = time.strftime("%Y%m%d-%H%M%S")
results_path = f"offline_results_{run_id}.csv"
predictions_path = f"offline_predictions_{run_id}.txt"
accuracy_path = f"offline_accuracy_{run_id}.txt"

# ---------- load data ----------
X = np.load("./inputs/matrices/mnist_test_images.npy", allow_pickle=True)
if X.ndim == 3 and X.shape[1:] == (28, 28):
    X = X.reshape(X.shape[0], -1)  # (N, 784)
Y = np.load("./inputs/matrices/mnist_test_labels.npy", allow_pickle=True)

# ---------- load model ----------
W1 = np.load("./inputs/matrices/W1.npy", allow_pickle=True)
B1 = np.load("./inputs/matrices/B1.npy", allow_pickle=True)
W2 = np.load("./inputs/matrices/W2.npy", allow_pickle=True)
B2 = np.load("./inputs/matrices/B2.npy", allow_pickle=True)

# Per-layer gains (to align scaled-domain outputs to bias units)
ALPHA_1 = 10.0
BETA_1 = 0.0
ALPHA_2 = 10.0
BETA_2 = 0.0