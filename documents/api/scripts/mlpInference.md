# Driver Script: `mlpInference.py`

**File:** `mlpInference.py`

---

## Overview

`mlpInference.py` runs a two-layer MLP (Multi-Layer Perceptron) neural network inference entirely on memristive crossbar hardware (or its simulation). It uses the MELISO+ framework for each matrix-vector multiply (MVM) and applies per-layer gain scaling to compensate for the hardware's normalised output domain.

The script is targeted at the MNIST digit classification task but is straightforwardly adaptable to other pre-trained models.

---

## Usage

```bash
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
  mpiexec -n 2 python3 mlpInference.py
```

---

## Functions

### `softmax(x)`

Numerically stable softmax.

```python
def softmax(x: np.ndarray) -> np.ndarray
```

```
z = x - max(x)
return exp(z) / sum(exp(z))
```

---

### `to_unit_interval(v, eps=1e-12)`

Rescale a non-negative vector to `[0, 1]` by dividing by its maximum.

```python
def to_unit_interval(v: np.ndarray, eps: float = 1e-12) -> np.ndarray
```

- Applies ReLU clamp (`max(v, 0)`) before scaling.
- Returns a zero vector if `max(v) < eps`.

Used to keep inter-layer activations in the hardware's operating range.

---

### `mem_mvm_scaled(x_scaled, W_scaled, y_path=None)`

Perform a single MVM on memristive hardware and return the scaled-domain output.

```python
def mem_mvm_scaled(
    x_scaled: np.ndarray,
    W_scaled: np.ndarray,
    y_path: str | None = None
) -> np.ndarray
```

| Parameter | Description |
|-----------|-------------|
| `x_scaled` | Input vector already scaled to `[0, 1]` |
| `W_scaled` | Weight matrix already scaled to `[0, 1]` |
| `y_path` | Optional override for the result file path |

Steps:

1. Creates a `MatVecSolver(xvec=x_scaled, mat=W_scaled)`.
2. Calls `initializeVec()` and `initializeMat()`.
3. Calls `matVec(correction=False)` — stays in scaled domain.
4. Reads the result from `<TMPDIR>/y_mem_result.txt`.
5. Calls `finalize()`.

**Returns:** 1-D NumPy array with the scaled MVM output.

---

### `run_layer_scaled(x_in_scaled, W_scaled, b, alpha=1.0, beta=1.0, relu=True)`

Run one complete linear layer including bias and optional ReLU, returning both the raw output and the rescaled activation for the next layer.

```python
def run_layer_scaled(
    x_in_scaled: np.ndarray,
    W_scaled: np.ndarray,
    b: np.ndarray,
    alpha: float = 1.0,
    beta: float = 1.0,
    relu: bool = True
) -> tuple[np.ndarray, np.ndarray]
```

| Parameter | Description |
|-----------|-------------|
| `x_in_scaled` | Input activation in `[0, 1]` |
| `W_scaled` | Weight matrix in `[0, 1]` |
| `b` | Bias vector |
| `alpha` | Per-layer gain to map scaled output back toward real-domain magnitude |
| `beta` | Additive shift applied alongside bias |
| `relu` | If `True`, apply ReLU activation |

Formula:

```
y_scaled = mem_mvm_scaled(x_in_scaled, W_scaled)
y        = alpha * y_scaled + b + beta
if relu:
    y = max(y, 0)
x_next = to_unit_interval(y)
```

**Returns:** `(y, x_next)` — the layer output and the normalised activation for the next layer.

---

## Script Configuration

### Data Paths

```python
X = np.load("./inputs/matrices/mnist_test_images.npy")  # shape (N, 784)
Y = np.load("./inputs/matrices/mnist_test_labels.npy")
W1 = np.load("./inputs/matrices/W1.npy")
B1 = np.load("./inputs/matrices/B1.npy")
W2 = np.load("./inputs/matrices/W2.npy")
B2 = np.load("./inputs/matrices/B2.npy")
```

### Per-Layer Gains

| Constant | Value | Description |
|----------|-------|-------------|
| `ALPHA_1` | `10.0` | Layer 1 output gain |
| `BETA_1` | `0.0` | Layer 1 shift |
| `ALPHA_2` | `10.0` | Layer 2 output gain |
| `BETA_2` | `0.0` | Layer 2 shift |

### Output Files

Files are named with a timestamp (`run_id = time.strftime("%Y%m%d-%H%M%S")`):

| File | Contents |
|------|----------|
| `offline_results_<run_id>.csv` | Per-sample results |
| `offline_predictions_<run_id>.txt` | Predicted class labels |
| `offline_accuracy_<run_id>.txt` | Classification accuracy |

---

## Network Architecture

```
Input x ∈ R^784  (MNIST flattened image)
   │
   ▼  W1 ∈ R^{hidden × 784}, B1 ∈ R^hidden
Layer 1: y1 = ALPHA_1 * (W1_scaled @ x_scaled) + B1 + BETA_1
         → ReLU → rescale to [0,1]
   │
   ▼  W2 ∈ R^{10 × hidden}, B2 ∈ R^10
Layer 2: y2 = ALPHA_2 * (W2_scaled @ x1_scaled) + B2 + BETA_2
         → softmax → predicted class = argmax
```

---

## Environment Variables

Inherits all variables from the MELISO+ framework:

| Variable | Description |
|----------|-------------|
| `EXP_CONFIG_FILE` | **Required.** Experiment YAML path |
| `DT` | Device type (default `1`) |
| `ITER_LIMIT` | Write-and-verify iteration cap |
| `TMPDIR` | Temp directory for intermediate files |

---

## See Also

- [`MatVecSolver`](../solver/matvec/MatVecSolver.md) — Used for each layer's MVM.
- [`DistributedMatVec.py`](DistributedMatVec.md) — Simpler MVM driver script.
