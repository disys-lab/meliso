# `MatVecSolver`

**Module:** `solver.matvec.MatVecSolver`
**File:** `solver/matvec/MatVecSolver.py`

---

## Overview

`MatVecSolver` is the primary user-facing interface for performing distributed matrix-vector multiplication (MVM) with MELISO+. It abstracts away MPI process roles, hardware initialisation, and the virtualization protocol, exposing a simple API:

```python
mv = MatVecSolver(xvec=x, mat=A)
mv.matVec(correction=True)
mv.finalize()
y = mv.acquireResults()
```

Internally, `MatVecSolver` creates either a [`Root`](Root.md) or [`NonRoot`](NonRoot.md) solver object depending on the MPI rank:

- **Root process** (`rank == size - 1`): creates a `Root` instance.
- **Worker processes** (`rank < size - 1`): create a `NonRoot` instance.

---

## Class Definition

```python
class MatVecSolver:
    def __init__(self, xvec=None, mat=None): ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `xvec` | `np.ndarray` or `None` | `None` | Input vector. If `None`, loaded from `XVEC_PATH` env var (default `inputs/vectors/input_x.txt`) |
| `mat` | `np.ndarray` or `None` | `None` | Weight matrix. If `None`, loaded from the path in the experiment YAML |

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `comm` | `MPI.Comm` | MPI communicator (`COMM_WORLD`) |
| `rank` | `int` | This process's MPI rank |
| `size` | `int` | Total number of MPI processes |
| `solverObject` | `Root` or `NonRoot` | Process-role object |
| `xvec` | `np.ndarray` | Input vector |
| `mat` | `np.ndarray` or `None` | Input matrix (may be `None` on non-root processes) |
| `y` | `np.ndarray` | MVM result; initialised to `np.zeros_like(xvec)` at construction, populated after `acquireResults()` |

---

## Methods

### `initializeVec()`

Explicitly initialise the input vector on the root process.

```python
def initializeVec(self) -> None
```

Only has effect on the root process. Calls `self.solverObject.initializeX(self.xvec)`.

---

### `initializeMat()`

Explicitly initialise the matrix on the root process.

```python
def initializeMat(self) -> None
```

Only has effect on the root process. Calls `self.solverObject.initializeMat(self.mat)`.

---

### `matVec(correction=False)`

Execute the distributed MVM.

```python
def matVec(self, correction: bool = False) -> None
```

| Parameter | Description |
|-----------|-------------|
| `correction` | If `True`, the root process reverses min-max scaling on the output, returning results in the original domain. |

This is a collective call — **all MPI processes must call it** (root coordinates, workers execute).

!!! note "Auto-initialization"
    As of the latest version, `matVec()` calls `initializeVec()` and `initializeMat()` internally before executing the MVM. Explicit prior calls to these methods are no longer required when using the standard workflow.

---

### `parallelizedBenchmarkMatVec(hardwareOn=1, scalingOn=0, correction=False)`

Run the MVM again in benchmark mode and compare memristive results against a CPU (NumPy) baseline.

```python
def parallelizedBenchmarkMatVec(
    self,
    hardwareOn: int = 1,
    scalingOn: int = 0,
    correction: bool = False
) -> None
```

Prints relative L2 and L∞ errors between the memristive and CPU results.

---

### `acquireMCAStats()`

Gather and print hardware performance statistics from all processes.

```python
def acquireMCAStats(self) -> None
```

Delegates to `solverObject.acquireMCAStats()`.

---

### `finalize()`

Signal worker processes to exit their instruction-wait loop.

```python
def finalize(self) -> None
```

Must be called after each `matVec()` or `parallelizedBenchmarkMatVec()` call to keep workers and root synchronised.

---

### `stopCommunication()`

Cleanly terminate the MPI environment.

```python
def stopCommunication(self) -> None
```

Calls `MPI.Finalize()`. **Must be called after all MPI computation is complete** and after `finalize()` has been called to ensure worker processes have exited their loops.

!!! warning
    Once `stopCommunication()` is called, no further MPI operations are possible. Call this only at the very end of your script.

---

### `acquireResults()`

Return the MVM output vector on the root process.

```python
def acquireResults(self) -> np.ndarray | None
```

Returns `None` on non-root processes. On the root process, returns `self.solverObject.y_mem_result`.

---

## Typical Usage Pattern

```python
from mpi4py import MPI
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

xvec = np.loadtxt("inputs/vectors/input_x.txt", delimiter=',')

# ── Run 1: without normalization reversal ──────────────────────────────────
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=False)
mv.finalize()
mv.acquireMCAStats()
y_scaled = mv.acquireResults()

mv.parallelizedBenchmarkMatVec(0, 0, correction=False)
mv.finalize()
mv.acquireMCAStats()

# ── Run 2: with normalization reversal ────────────────────────────────────
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=True)
mv.finalize()
mv.acquireMCAStats()
y_original_domain = mv.acquireResults()

# ── Clean up MPI at the very end ──────────────────────────────────────────
mv.stopCommunication()
```

---

## MPI Process Count

The number of MPI processes must be exactly `(mca_rows × mca_cols) + 1`:

- `mca_rows × mca_cols` worker processes (one per hardware array).
- `1` root process.

Example: `mca_rows=2, mca_cols=2` → launch with `mpiexec -n 5`.

---

## See Also

- [`Root`](Root.md) — Root-process coordinator.
- [`NonRoot`](NonRoot.md) — Worker-process coordinator.
- [`DistributedMatVec.py`](../../scripts/DistributedMatVec.md) — Basic MVM driver script.
- [`mlpInference.py`](../../scripts/mlpInference.md) — MLP inference driver.
- [`pdhg.py`](../../scripts/pdhg.md) — PDHG optimization driver.
