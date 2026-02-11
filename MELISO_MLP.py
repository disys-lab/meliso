#===================================================================================================
# OLD STABLE VERSION
#===================================================================================================
# import time, os
# import numpy as np
# from solver.matvec.MatVecSolver import MatVecSolver

# # ---------- numerically-stable softmax ----------
# def softmax(x):
#     z = x - np.max(x)
#     e = np.exp(z)
#     return e / np.sum(e)

# # ---------- keep next-layer inputs in [0,1] ----------
# def to_unit_interval(v, eps=1e-12):
#     v = np.asarray(v, dtype=np.float64)
#     v = np.clip(v, 0.0, None)       # ReLU outputs are nonnegative; clamp just in case
#     vmax = np.max(v)
#     if vmax < eps:
#         return np.zeros_like(v)
#     return v / vmax

# # ---------- one memristor MVM that returns SCALED output ----------
# def mem_mvm_scaled(x_scaled, W_scaled, y_path=None):
#     mv = MatVecSolver(xvec=x_scaled, mat=W_scaled)
#     mv.initializeVec()
#     mv.initializeMat()
#     # IMPORTANT: stay in scaled domain (no inverse/unscale)
#     mv.matVec(correction=False)
#     if y_path is None:
#         base = os.environ.get("TMPDIR", ".")
#         y_path = os.path.join(base, "y_mem_result.txt")
#         y_scaled = np.loadtxt(y_path)   # robust: default whitespace delimiter
#         if y_scaled.ndim > 1:
#             y_scaled = y_scaled[:, 0]
#         mv.finalize()
#     return y_scaled

# # ---------- one linear layer run in scaled domain ----------
# def run_layer_scaled(x_in_scaled, W_scaled, b, alpha=1.0, beta=1.0,relu=True):
#     """
#     """
#     y_scaled = mem_mvm_scaled(x_in_scaled, W_scaled)  # scaled-domain MVM result
#     y = alpha * y_scaled + b + beta                   # add bias in real domain
#     if relu:
#         y = np.maximum(y, 0.0)
#     x_next = to_unit_interval(y)                      # keep next layer input in [0,1]
#     return y, x_next

# # ---------- paths ----------
# run_id = time.strftime("%Y%m%d-%H%M%S")
# results_path = f"offline_results_{run_id}.csv"
# predictions_path = f"offline_predictions_{run_id}.txt"
# accuracy_path = f"offline_accuracy_{run_id}.txt"

# # ---------- load data ----------
# X = np.load("./inputs/matrices/mnist_test_images.npy", allow_pickle=True)
# if X.ndim == 3 and X.shape[1:] == (28, 28):
#     X = X.reshape(X.shape[0], -1)  # (N, 784)
# Y = np.load("./inputs/matrices/mnist_test_labels.npy", allow_pickle=True)

# # ---------- load model ----------
# W1 = np.load("./inputs/matrices/W1.npy", allow_pickle=True)
# B1 = np.load("./inputs/matrices/B1.npy", allow_pickle=True)
# W2 = np.load("./inputs/matrices/W2.npy", allow_pickle=True)
# B2 = np.load("./inputs/matrices/B2.npy", allow_pickle=True)

# # Per-layer gains (to align scaled-domain outputs to bias units)
# ALPHA_1 = 10.0
# BETA_1 = 0.0
# ALPHA_2 = 10.0
# BETA_2 = 0.0

#===================================================================================================
# NEW VERSION WITH CALIBRATION
#===================================================================================================
import time, os
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

# ======= simple flags for calibration =======
CALIBRATE  = True       # set False to use the fixed ALPHA_*/BETA_* below
CALIB_K    = 128        # how many samples for calibration (uses first K images)
USE_OFFSET = True       # learn beta (offset) too; set False for gain-only
# ============================================

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
    mv.matVec(correction=False)  # stay in scaled domain

    # always resolve y_path (default to TMPDIR)
    if y_path is None:
        base = os.environ.get("TMPDIR", ".")
        y_path = os.path.join(base, "y_mem_result.txt")

    y_scaled = np.loadtxt(y_path)   # robust: default whitespace delimiter
    if y_scaled.ndim > 1:
        y_scaled = y_scaled[:, 0]
    mv.finalize()
    return y_scaled

# ---------- one linear layer run in scaled domain ----------
def run_layer_scaled(x_in_scaled, W_scaled, b, alpha=1.0, beta=0.0, relu=True):
    """
    alpha, beta can be scalars or (m,) vectors
    """
    y_scaled = mem_mvm_scaled(x_in_scaled, W_scaled)  # scaled-domain MVM result
    y = alpha * y_scaled + b + beta                   # add bias in float domain
    if relu:
        y = np.maximum(y, 0.0)
    x_next = to_unit_interval(y)                      # keep next layer input in [0,1]
    return y, x_next

# ======= calibration helper (per-output alpha, optional beta) =======
def calibrate_alpha_beta(W_scaled, X_calib_scaled, use_offset=True, eps=1e-12):
    """
    For each output j: solve y_ref = alpha_j * y_sc (+ beta_j),
    where y_ref = W_scaled @ x (float path) and y_sc comes from mem_mvm_scaled.
    """
    Xc = np.asarray(X_calib_scaled, dtype=np.float64)
    K, d = Xc.shape
    m, dW = W_scaled.shape
    assert d == dW, "Calibration X and W have incompatible shapes."

    Y_ref = np.empty((K, m), dtype=np.float64)
    Y_sc  = np.empty((K, m), dtype=np.float64)

    for i in range(K):
        x = Xc[i]
        Y_ref[i] = W_scaled @ x
        Y_sc[i]  = mem_mvm_scaled(x, W_scaled)

    if use_offset:
        # closed-form least squares for [alpha; beta]
        S_ys = np.sum(Y_sc * Y_ref, axis=0)     # Sigma y_sc*y_ref
        S_yy = np.sum(Y_sc * Y_sc,  axis=0)     # Sigma y_sc^2
        S_y  = np.sum(Y_sc,         axis=0)     # Sigma y_sc
        S_r  = np.sum(Y_ref,        axis=0)     # Sigma y_ref
        Kf   = float(K)

        den   = (S_yy * Kf) - (S_y ** 2)
        den   = np.where(np.abs(den) < eps, np.inf, den)
        num_a = (S_ys * Kf) - (S_y * S_r)
        num_b = (S_r * S_yy) - (S_y * S_ys)
        alpha = num_a / den
        beta  = num_b / den
    else:
        den   = np.sum(Y_sc * Y_sc, axis=0)
        den   = np.where(np.abs(den) < eps, np.inf, den)
        alpha = np.sum(Y_sc * Y_ref, axis=0) / den
        beta  = np.zeros_like(alpha)

    # guard against NaN/Inf
    alpha = np.where(np.isfinite(alpha), alpha, 1.0)
    beta  = np.where(np.isfinite(beta),  beta,  0.0)
    return alpha, beta
# ==============================================================

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

# Initial per-layer gains (will be overridden if CALIBRATE=True)
ALPHA_1 = 10.0
BETA_1  = 0.0
ALPHA_2 = 10.0
BETA_2  = 0.0

# ======= optional calibration prepass (tiny and fast) =======
if CALIBRATE:
    K = min(CALIB_K, X.shape[0])

    # Layer-1 calibration inputs must match inference preprocessing
    X1_calib = np.array([to_unit_interval(x.astype(np.float64)) for x in X[:K]])
    a1, b1   = calibrate_alpha_beta(W1, X1_calib, use_offset=USE_OFFSET)

    # Build hidden calibration batch for layer 2 along the *float* path that mirrors inference
    H = np.maximum(X1_calib @ W1.T + B1, 0.0)
    H = np.array([to_unit_interval(h) for h in H])  # bound_next=True in inference
    a2, b2 = calibrate_alpha_beta(W2, H, use_offset=USE_OFFSET)

    # override per-layer gains
    ALPHA_1, BETA_1 = a1, b1
    ALPHA_2, BETA_2 = a2, b2

    with open("calibration.txt", "w+") as f1:
        f1.write(f"{a1}, {b1}, {a2}, {b2}\n")
        f1.close()
# ============================================================
else:
    a1, b1, a2, b2 = np.loadtxt("calibration.txt", delimiter=",").flatten()
    ALPHA_1, BETA_1 = a1, b1
    ALPHA_2, BETA_2 = a2, b2

    num_images = X.shape[0]
    correct = 0

    with open(results_path, "w+") as f:
        f.write("image_idx,pred,true,correct\n")
    with open(predictions_path, "w+") as f:
        f.write("")

    for i in range(num_images):
        #=============================================================================================
        # MODEL INFERENCE FOR IMAGE
        #=============================================================================================

        # --- Prepare input to be in [0,1] ---
        x0 = X[i].astype(np.float64)
        x1_in = to_unit_interval(x0)

        # --- Layer 1 (scaled domain) ---
        y1, x2_in = run_layer_scaled(x1_in, W1, B1, alpha=ALPHA_1, beta=BETA_1, relu=True)

        # --- Layer 2 (scaled domain) ---
        y2, _ = run_layer_scaled(x2_in, W2, B2, alpha=ALPHA_2, beta=BETA_2, relu=False)

        # --- Softmax & bookkeeping ---
        probs = softmax(y2)
        pred = int(np.argmax(probs))
        true = int(Y[i])

        with open(predictions_path, "a") as f1:
            f1.write(f"{i}: {probs}\n")
        with open(results_path, "a") as f2:
            f2.write(f"{i},{pred},{true},{int(pred==true)}\n")

        if pred == true:
            correct += 1
    f1.close()
    f2.close()
    acc = correct / num_images
    with open(accuracy_path, "w+") as f3:
        f3.write(f"Offline Accuracy: {acc * 100:.2f}%\n")
        f3.write(f"Evaluated {num_images} images\n")
        f3.close()
    print(f"Accuracy on {num_images} images: {acc*100:.2f}%")
