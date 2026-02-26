# `RootMCA`

**Module:** `src.core.RootMCA`
**File:** `src/core/RootMCA.py`
**Inherits:** [`BaseMCA`](BaseMCA.md)

---

## Overview

`RootMCA` is the coordinator process class responsible for:

1. Loading and min-max scaling the input matrix.
2. Zero-padding the matrix to fit the MCA grid dimensions.
3. Decomposing the padded matrix into chunks and distributing them to worker processes via MPI `Send`.
4. Sending slices of the input vector to workers.
5. Receiving partial results from workers and summing them into the final output vector.
6. Gathering and printing hardware performance statistics from all processes.
7. Writing experiment configuration and statistics to a report file.

The root process occupies MPI rank `size - 1`.

---

## Class Definition

```python
class RootMCA(BaseMCA):
    def __init__(self, comm): ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `comm` | `mpi4py.MPI.Comm` | MPI communicator |

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `col_parts` | `dict[int, [int, int]]` | Maps each worker rank to `[col_start, col_end]` in the padded matrix |
| `row_parts_ranks` | `dict[int, list[int]]` | Maps each row-start index to the list of ranks that hold that row block |
| `matRows` | `int` | Padded matrix row count |
| `matCols` | `int` | Padded matrix column count |
| `origMatRows` | `int` | Original matrix row count (before padding) |
| `origMatCols` | `int` | Original matrix column count (before padding) |
| `mat` | `np.ndarray` | Scaled (and padded) matrix |
| `globalMat` | `np.ndarray` | Original unscaled matrix |
| `x` | `np.ndarray` | (Possibly padded) input vector |
| `globalX` | `np.ndarray` | Original input vector |
| `mat_min` | `float` | Global minimum of the matrix (used for scaling reversal) |
| `mat_max` | `float` | Peak-to-peak range of the matrix (used for scaling reversal) |
| `mat_row_sum` | `np.ndarray` | Row sums of the original matrix |
| `x_min` | `float` | Minimum of input vector |
| `x_max` | `float` | Peak-to-peak range of input vector |
| `x_sum` | `float` | Sum of input vector elements |
| `allMCAStats` | `np.ndarray` | Shape `(size, 8, 1)` — gathered stats from all processes |
| `hardwareOn` | `int` | `1` if hardware simulation is active |
| `scalingOn` | `int` | `1` if scaling is active |

---

## Methods

### `printConfiguration()`

Prints and writes to report the current experiment configuration from `self.exp_config["exp_params"]`.

```python
def printConfiguration(self) -> None
```

---

### `initializeMatrix(mat)`

Entry point for loading the matrix.

```python
def initializeMatrix(self, mat: np.ndarray | None) -> None
```

- If `mat` is `None`, reads the matrix from the path specified by `matrix_file` in the YAML.
- If `mat` is provided, uses it directly.
- Calls `scaleMatrix()` in both cases and stores scaling parameters.

---

### `processMatrixFile()`

Delegates to `readMatrix(self.matrix_file)`.

---

### `readMatrix(filename)`

Reads a matrix from disk.

```python
def readMatrix(self, filename: str) -> None
```

**Supported formats:**

| Extension | Loader |
|-----------|--------|
| `.mtx` | `scipy.io.mmread` → `.toarray()` |
| `.npy` | `numpy.load` |
| `.csv` | `numpy.loadtxt(delimiter=',')` |
| `.txt` | `numpy.loadtxt(delimiter=',')` |

Sets `self.mat`, `self.globalMat`, `self.origMatRows`, `self.origMatCols`.

**Raises:** `Exception` if the file does not exist or the format is unsupported.

---

### `setMat(mat)`

Pads the matrix and distributes chunks to worker processes.

```python
def setMat(self, mat: np.ndarray) -> None
```

Steps:

1. `padMatrix(mat)` — zero-pads rows/columns to align with the MCA grid.
2. `createDecompositionDir()` — creates the temp directory for chunk files.
3. `distributeMatrixChunksFileWrite()` — writes chunks to disk and sends via MPI.

---

### `setX(x)`

Validates and optionally zero-pads the input vector to match the padded matrix column count.

```python
def setX(self, x: np.ndarray) -> None
```

**Raises:** `Exception` if `x` is longer than the padded matrix column count.

---

### `scaleMatrix(mat)`

Applies global min-max normalization to `mat`, mapping values to `[0, 1]`.

```python
def scaleMatrix(self, mat: np.ndarray) -> tuple[np.ndarray, float, float, np.ndarray]
```

**Returns:** `(scaled_mat, mat_min, mat_max_range, mat_row_sum)`

Formula:

```
scaled = (mat - mat.min()) / mat.ptp()
```

---

### `padMatrix(mat)`

Zero-pads `mat` so that its dimensions are exact multiples of `(mcaRows * cellRows, mcaCols * cellCols)`.

```python
def padMatrix(self, mat: np.ndarray) -> tuple[np.ndarray, int, int]
```

**Returns:** `(padded_mat, new_rows, new_cols)`

**Raises:** `Exception` if the MCA grid would exceed the matrix dimensions (i.e., `mca_rows * cell_rows - rows < 0`).

**Warnings:** Prints a message if the MCA grid is larger than necessary.

---

### `distributeMatrixChunksFileWrite()`

Iterates over all `(i, j)` positions in the MCA grid, extracts the corresponding sub-matrix, saves it to `<decomp_dir>/<i>_<j>.npy`, and sends it to rank `i * mcaCols + j` via `comm.Send`.

Also populates `self.col_parts` and `self.row_parts_ranks`.

---

### `position_assign(P, Q, M, N, R, C, i, j)`

Computes the row/column slice boundaries for the `(i, j)`-th MCA chip.

```python
def position_assign(
    self,
    P: int,   # cellRows
    Q: int,   # cellCols
    M: int,   # mcaRows
    N: int,   # mcaCols
    R: int,   # total matrix rows (padded)
    C: int,   # total matrix cols (padded)
    i: int,   # row index in the MCA grid
    j: int    # col index in the MCA grid
) -> tuple[int, int, int, int]   # (s_c, e_c, s_r, e_r)
```

**Mapping strategy:** `Matrix (R×C) → (M*P) × (N*Q)`

Assumes `R == M * P` and `C == N * Q`.

---

### `parallelMatVec()`

Orchestrates one round of distributed MVM:

1. Sends the appropriate slice of `self.x` to every worker rank.
2. Receives the partial output vector `y` from every worker rank.
3. Accumulates partial results by row.
4. Returns `sum_y[:origMatRows]` (strips padding).

```python
def parallelMatVec(self) -> np.ndarray
```

**Returns:** Output vector of shape `(origMatRows,)`.

---

### `getMCAStats()`

Gathers MCA statistics from all processes via `comm.Gather`, prints per-rank and aggregate stats (write/read latency and energy), and writes them to the report file.

```python
def getMCAStats(self) -> None
```

**Statistics collected (8 per rank):**

| Index | Statistic |
|-------|-----------|
| 0 | `totalSubArrayArea` |
| 1 | `totalNeuronAreaIH` |
| 2 | `subArrayIHLeakage` |
| 3 | `leakageNeuronIH` |
| 4 | `subArrayIH→writeLatency` |
| 5 | `arrayIH→writeEnergy + subArrayIH→writeDynamicEnergy` |
| 6 | `subArrayIH→readLatency` |
| 7 | `arrayIH→readEnergy + subArrayIH→readDynamicEnergy` |

---

## Module-Level Utilities

### `__report_path__(fallback_name)`

Returns a `Path` for the report file. Uses the `REPORT_PATH` environment variable if set, otherwise falls back to `fallback_name`. Creates parent directories as needed.

### `__write_report__(content)`

Appends a string to the report file (as determined by `__report_path__()`).

---

## See Also

- [`BaseMCA`](BaseMCA.md) — Base class with config parsing.
- [`NonRootMCA`](NonRootMCA.md) — Counterpart worker process class.
- [`Root`](../../solver/matvec/Root.md) — Higher-level coordinator that wraps `RootMCA`.
