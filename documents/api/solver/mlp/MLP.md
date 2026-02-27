# `MLP`

**Module:** `solver.mlp.MLP`
**File:** `solver/mlp/MLP.py`

---

## Overview

`MLP` implements a two-layer multi-layer perceptron (MLP) for MNIST digit classification. It is used by [`mlpInference.py`](../../scripts/mlpInference.md) to provide CPU-reference inference results and to supply the trained weight matrices to the MELISO+ accelerated inference pipeline.

The network architecture is:

```
Input x ∈ R^784   (flattened 28×28 MNIST image)
   │
   ▼  W1 ∈ R^{512 × 784},  B1 ∈ R^512,  fan-in = 784
Layer 1:  z1 = (W1 @ x) / FANIN_1 + B1
          a1 = clip(z1, 0, 1)           ← clipped ReLU
   │
   ▼  W2 ∈ R^{10 × 512},  B2 ∈ R^10,  fan-in = 512
Layer 2:  z2 = (W2 @ a1) / FANIN_2 + B2
          a2 = softmax(temperature × z2)
   │
   ▼
Predicted class = argmax(a2)
```

---

## Class Definition

```python
class MLP:
    def __init__(self, W1_path, B1_path, W2_path, B2_path): ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `W1_path` | `str` | Path to first-layer weight matrix `.npy` file |
| `B1_path` | `str` | Path to first-layer bias vector `.npy` file |
| `W2_path` | `str` | Path to second-layer weight matrix `.npy` file |
| `B2_path` | `str` | Path to second-layer bias vector `.npy` file |

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `W1` | `np.ndarray` | First-layer weight matrix, shape `(hidden, 784)` |
| `B1` | `np.ndarray` | First-layer bias vector, shape `(hidden,)` |
| `W2` | `np.ndarray` | Second-layer weight matrix, shape `(10, hidden)` |
| `B2` | `np.ndarray` | Second-layer bias vector, shape `(10,)` |
| `FANIN_1` | `float` | First-layer fan-in normalisation constant (`784.0`) |
| `FANIN_2` | `float` | Second-layer fan-in normalisation constant (`512.0`) |
| `temperature` | `float` | Softmax temperature scaling factor (`1.0`) |

---

## Methods

### `predict(input_vector)`

Run a full forward pass through both layers and return all intermediate activations.

```python
def predict(self, input_vector) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `input_vector` | array-like | Input sample; can be a list, 1-D array `(784,)`, or 2-D array `(28, 28)` |

**Returns:** `(z1, z2, a1, a2)` where:

| Return value | Shape | Description |
|--------------|-------|-------------|
| `z1` | `(hidden,)` | Pre-activation output of layer 1 |
| `z2` | `(10,)` | Pre-activation output of layer 2 |
| `a1` | `(hidden,)` | Post-activation output of layer 1 (clipped ReLU) |
| `a2` | `(10,)` | Post-activation output of layer 2 (softmax probabilities) |

**Raises:** `ValueError` if `input_vector.size != W1.shape[1]`.

#### Forward Pass Detail

```python
# Layer 1
z1 = (W1 @ input_vector / FANIN_1) + B1
a1 = np.clip(z1, 0.0, 1.0)          # Clipped ReLU: clips to [0, 1]

# Layer 2
z2 = (W2 @ a1 / FANIN_2) + B2
a2 = softmax(temperature * z2)
```

!!! note "Clipped ReLU vs Standard ReLU"
    Layer 1 uses `np.clip(z1, 0.0, 1.0)` (clipping to `[0, 1]`), not a standard ReLU. This keeps activations within the hardware's operating range when used with MELISO+ in subsequent layers.

---

### `__relu__(x)` (private)

Standard ReLU activation: `max(0, x)`.

```python
def __relu__(self, x) -> np.ndarray
```

!!! note
    `__relu__` is not used internally by `predict()`. It is exposed for external use, e.g. in `mlpInference.py` where it is called explicitly after an MELISO+ MVM.

---

### `__softmax__(x)` (private)

Numerically stable softmax.

```python
def __softmax__(self, x) -> np.ndarray
```

```
z = x - max(x)
return exp(z) / sum(exp(z))
```

---

### `__load_model__(W1_path, B1_path, W2_path, B2_path)` (private)

Load all model parameters from `.npy` files.

```python
def __load_model__(self, W1_path, B1_path, W2_path, B2_path)
    -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
```

Loads files with `np.load(..., allow_pickle=True)` and prints the loaded shapes for validation.

---

## Typical Usage

```python
from solver.mlp.MLP import MLP
import numpy as np

# Load model
model = MLP(
    W1_path="./inputs/mlp/W1.npy",
    B1_path="./inputs/mlp/B1.npy",
    W2_path="./inputs/mlp/W2.npy",
    B2_path="./inputs/mlp/B2.npy"
)

# Run inference on a single MNIST sample
test_images = np.load("./inputs/mlp/mnist_test_images.npy")  # shape (10000, 784)
input_vector = test_images[0]

z1, z2, a1, a2 = model.predict(input_vector)
predicted_class = int(np.argmax(a2))
print(f"Predicted: {predicted_class}")
```

---

## Expected Input Files

| File | Shape | Description |
|------|-------|-------------|
| `W1.npy` | `(512, 784)` | Layer 1 weight matrix |
| `B1.npy` | `(512,)` | Layer 1 bias vector |
| `W2.npy` | `(10, 512)` | Layer 2 weight matrix |
| `B2.npy` | `(10,)` | Layer 2 bias vector |

These files should be stored under `./inputs/mlp/` relative to the working directory.

---

## See Also

- [`mlpInference.py`](../../scripts/mlpInference.md) — Driver script that uses `MLP` for both CPU and MELISO+ accelerated inference.
- [`MatVecSolver`](../matvec/MatVecSolver.md) — Used in `mlpInference.py` to accelerate layer MVM on hardware.
