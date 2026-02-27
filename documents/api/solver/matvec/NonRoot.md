# `NonRoot`

**Module:** `solver.matvec.NonRoot`
**File:** `solver/matvec/NonRoot.py`

---

## Overview

`NonRoot` is the high-level coordinator for worker MPI processes (ranks `0` through `size - 2`). It wraps [`NonRootMCA`](../../src/core/NonRootMCA.md) and adds the **virtualization protocol** — a loop that awaits tile-coordinate broadcasts from the root and processes each tile in turn.

---

## Class Definition

```python
class NonRoot:
    def __init__(self, comm, verbose=False): ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comm` | `mpi4py.MPI.Comm` | — | MPI communicator |
| `verbose` | `bool` | `False` | If `True`, prints diagnostic messages for each broadcast received, tile assignment, and MVM step |

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `comm` | `MPI.Comm` | MPI communicator |
| `mca` | `NonRootMCA` | Lower-level worker instance (created with `set_mat=False`) |
| `virtualizationOn` | `bool` | `True` by default — enables the tile-wait loop |
| `verbose` | `bool` | Enables verbose diagnostic output when `True` |
| `y` | `np.ndarray` | Local output vector from the most recent MVM |

---

## Methods

### `awaitInstructions()`

Blocks until the root broadcasts a tile-coordinate pair or the stop sentinel.

```python
def awaitInstructions(self) -> bool
```

Protocol:

1. Calls `comm.Bcast(data, root=ROOT_PROCESS_RANK)` to receive a `float64[2]` array.
2. If `data[0] >= 0`: tile coordinates are valid → calls `mca.setMat()` to receive and encode the tile, returns `True`.
3. If `data[0] < 0`: stop sentinel (`[-1, -1]`) → returns `False`.

---

### `parallelMatVec(correction=False)`

Execute the worker-side MVM loop.

```python
def parallelMatVec(self, correction: bool = False) -> None
```

- If `virtualizationOn=True`: repeatedly calls `awaitInstructions()` until it returns `False`, executing `mca.parallelMatVec()` for each accepted tile.
- If `virtualizationOn=False`: calls `mca.parallelMatVec()` once.

The `correction` parameter is accepted for API compatibility but has no effect on the worker side (scaling reversal is handled by the root).

---

### `benchmarkMatVecParallel(hardwareOn=0, scalingOn=0, correction=False)`

Run a benchmark pass by temporarily overriding hardware and scaling flags.

```python
def benchmarkMatVecParallel(
    self,
    hardwareOn: int = 0,
    scalingOn: int = 0,
    correction: bool = False
) -> None
```

Sets `mca.meliso_obj.setHardwareOn(hardwareOn)` and `setScalingOn(scalingOn)`, then calls `parallelMatVec(correction)`.

---

### `benchmarkMatVec()`

Placeholder; no-op.

---

### `acquireMCAStats()`

Delegates to `mca.getMCAStats()`.

```python
def acquireMCAStats(self) -> None
```

---

### `finalize()`

No-op on worker processes. The root calls `Root.finalize()` which broadcasts the stop sentinel that terminates the worker's `awaitInstructions` loop.

```python
def finalize(self) -> None
```

---

## Tile-Processing Protocol

```
Worker                          Root
  │                               │
  │ ◄── Bcast([i, j]) ────────────│  (virtualParallelMatVec)
  │                               │
  │  setMat()                     │
  │   Recv(matrix_chunk) ◄────────│  (Send from RootMCA)
  │   initializeMCA()             │
  │                               │
  │  parallelMatVec()             │
  │   Recv(x_chunk) ◄─────────────│
  │   localMatVec()               │
  │   Send(y_chunk) ──────────────►
  │                               │
  │ ◄── Bcast([-1, -1]) ──────────│  (finalize)
  │  (exits loop)                 │
```

---

## See Also

- [`NonRootMCA`](../../src/core/NonRootMCA.md) — Lower-level worker class used internally.
- [`Root`](Root.md) — Root counterpart.
- [`MatVecSolver`](MatVecSolver.md) — High-level interface that creates `NonRoot`.
