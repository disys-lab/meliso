# Driver Script: `mlpInference.py`

**File:** `mlpInference.py`
**Authors:** Huynh Quang Nguyen Vo (Oklahoma State University)

---

## Overview

`mlpInference.py` demonstrates inference with a pre-trained two-layer MLP on the MNIST digit classification task. It runs two inference passes:

1. **CPU inference** using the [`MLP`](../solver/mlp/MLP.md) class and standard NumPy.
2. **MELISO+ accelerated inference** using [`MatVecSolver`](../solver/matvec/MatVecSolver.md) to offload the first layer's matrix-vector multiplication to memristive crossbar hardware.

The second layer's MELISO+ acceleration is not yet fully implemented (see the `TODO` comment in the source).

---

## Usage

```bash
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
  mpiexec -n 2 python3 mlpInference.py
```

---

## Functions

### `cancel_SLURM_job()`

Attempt to cancel the current SLURM job if running in a SLURM environment.

```python
def cancel_SLURM_job() -> None
```

Reads the `SLURM_JOB_ID` environment variable and calls `scancel <job_id>` via `subprocess.run`. If `SLURM_JOB_ID` is not set, prints a warning and takes no action.

---

### `main()`

Main entry point. Loads the model and data, then runs CPU and MELISO+ inference.

```python
def main() -> None
```

---

## Script Flow

### 1. Load Model and Data

```python
model = MLP(
    W1_path="./inputs/mlp/W1.npy",
    B1_path="./inputs/mlp/B1.npy",
    W2_path="./inputs/mlp/W2.npy",
    B2_path="./inputs/mlp/B2.npy"
)

test_images = np.load("./inputs/mlp/mnist_test_images.npy")  # shape (10000, 784)
test_labels = np.load("./inputs/mlp/mnist_test_labels.npy")  # shape (10000,)
```

### 2. CPU Inference

Iterates over `subset_size` samples from the test set:

```python
subset_size = 1   # Adjust to evaluate more samples
for i in range(subset_size):
    z1_cpu, z2_cpu, a1_cpu, a2_cpu = model.predict(test_images[i])
    predicted_label = int(np.argmax(a2_cpu))
```

Reports accuracy percentage on the evaluated subset.

### 3. MELISO+ Accelerated Inference (Layer 1)

For each sample, the first layer's MVM is offloaded to MELISO+:

```python
CORRECTION = False  # Keep result in [0,1] (min-max scaled domain)
mv = MatVecSolver(xvec=input_vector, mat=model.W1)

mv.matVec(correction=CORRECTION)
mv.finalize()
mv.acquireMCAStats()
z1 = mv.acquireResults()      # Shape: (512, 1)
```

After acquiring `z1`, bias is added and ReLU is applied:

```python
z1 = z1.reshape(-1, 1)        # Ensure (512, 1) shape
z1 = z1 + model.B1            # Add bias
a1 = model.__relu__(z1)       # Apply ReLU activation
```

The script also prints the **relative L2 error** of both `z1` and `a1` against the CPU reference values.

### 4. Second Layer (Not Yet Implemented)

The second layer MELISO+ acceleration is commented out with a `TODO` note. The recommended approach is:

```python
# mv = MatVecSolver(xvec=a1.reshape(-1,1), mat=model.W2)
# mv.matVec(correction=CORRECTION)
# mv.finalize()
# mv.acquireMCAStats()
# z2 = mv.acquireResults()
# mv.stopCommunication()
```

---

## Configuration Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `subset_size` | `1` | Number of MNIST test samples to evaluate |
| `CORRECTION` | `False` | Whether to apply min-max scaling reversal (`False` = results in `[0,1]` range) |

---

## Network Architecture

```
Input x ∈ R^784   (MNIST flattened image, shape (784, 1))
   │
   ▼  W1 ∈ R^{512 × 784}   (distributed MVM via MELISO+)
Layer 1:  z1 = MVM(W1, x)        via MatVecSolver
           z1 = z1 + B1           add bias
           a1 = ReLU(z1)          shape (512, 1)
   │
   ▼  W2 ∈ R^{10 × 512}   (TODO: MELISO+ acceleration pending)
Layer 2:  z2 = W2 @ a1 / FANIN_2 + B2
           a2 = softmax(z2)       shape (10,)
   │
   ▼
Predicted class = argmax(a2)
```

---

## Input Files

All model and data files should be placed under `./inputs/mlp/`:

| File | Shape | Description |
|------|-------|-------------|
| `W1.npy` | `(512, 784)` | Layer 1 weight matrix |
| `B1.npy` | `(512,)` | Layer 1 bias vector |
| `W2.npy` | `(10, 512)` | Layer 2 weight matrix |
| `B2.npy` | `(10,)` | Layer 2 bias vector |
| `mnist_test_images.npy` | `(10000, 784)` | MNIST test images |
| `mnist_test_labels.npy` | `(10000,)` | MNIST test labels |

---

## Environment Variables

Inherits all MELISO+ environment variables:

| Variable | Description |
|----------|-------------|
| `EXP_CONFIG_FILE` | **Required.** Experiment YAML configuration path |
| `DT` | Device type (0–6); defaults to `1` (RealDevice) |
| `ITER_LIMIT` | Write-and-verify iteration cap |
| `TMPDIR` | Temporary directory for intermediate MVM files |
| `SLURM_JOB_ID` | If set, `cancel_SLURM_job()` will cancel this job |

---

## Output

- **Console:** Per-sample predictions, relative L2 errors, and accuracy percentage.
- **`<TMPDIR>/y_mem_result.txt`:** Intermediate MVM result from the MELISO+ accelerator.

---

## Dependencies

| Package | Usage |
|---------|-------|
| `meliso` | Compiled Cython extension |
| `solver.matvec.MatVecSolver` | Distributed MVM interface |
| `solver.mlp.MLP` | MLP model inference |
| `numpy` | Numerical operations |
| `subprocess` | SLURM job cancellation |

---

## See Also

- [`MLP`](../solver/mlp/MLP.md) — The MLP class used for both CPU inference and weight loading.
- [`MatVecSolver`](../solver/matvec/MatVecSolver.md) — The distributed MVM interface.
- [`DistributedMatVec.py`](DistributedMatVec.md) — Simpler single-MVM driver script.
- [`pdhg.py`](pdhg.md) — PDHG optimization driver using the same MELISO+ framework.
