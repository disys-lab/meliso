# `Root`

**Module:** `solver.matvec.Root`
**File:** `solver/matvec/Root.py`

---

## Overview

`Root` is the high-level coordinator for the root MPI process (rank `size - 1`). It wraps [`RootMCA`](../../src/core/RootMCA.md) and adds:

- **Virtualization**: automatically tiles matrices that are larger than the MCA hardware capacity, sequentially processing each tile.
- **Min-max scaling reversal**: optionally un-normalises the output vector back to the original domain after MVM.
- **Benchmarking**: compares memristive MVM results against a CPU (NumPy) baseline with relative L2 and L∞ error reporting.

---

## Class Definition

```python
class Root:
    def __init__(self, comm, x=None, mat=None): ...
```

### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `comm` | `mpi4py.MPI.Comm` | — | MPI communicator |
| `x` | `np.ndarray` or `None` | `None` | Input vector |
| `mat` | `np.ndarray` or `None` | `None` | Weight matrix (loaded from YAML if `None`) |

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `comm` | `MPI.Comm` | MPI communicator |
| `mca` | `RootMCA` | Lower-level coordinator instance |
| `deviceType` | `int` | Device type from `DT` env var |
| `hardwareOn` | `int` or `None` | Hardware simulation flag |
| `scalingOn` | `int` or `None` | Internal scaling flag |
| `origMat` | `np.ndarray` | Original (scaled) matrix |
| `origMatRows` | `int` | Original matrix row count |
| `origMatCols` | `int` | Original matrix column count |
| `cellRows` | `int` | Cells per MCA row |
| `cellCols` | `int` | Cells per MCA column |
| `mcaRows` | `int` | MCA grid rows |
| `mcaCols` | `int` | MCA grid columns |
| `mcaGridRowCap` | `int` | `mcaRows * cellRows` — total row capacity of the MCA grid |
| `mcaGridColCap` | `int` | `mcaCols * cellCols` — total column capacity |
| `maxVRows` | `int` | Number of virtual row tiles: `ceil(origMatRows / mcaGridRowCap)` |
| `maxVCols` | `int` | Number of virtual column tiles: `ceil(origMatCols / mcaGridColCap)` |
| `x` | `np.ndarray` | Scaled input vector |
| `globalX` | `np.ndarray` | Original (unscaled) input vector |
| `x_min` | `float` | Minimum of input vector |
| `x_max` | `float` | Range of input vector |
| `x_sum` | `float` | Sum of original input vector elements |
| `virtualizer` | `dict` | Tile metadata indexed by `(i, j)` or `i` |
| `virtualizationOn` | `bool` | `True` by default |
| `y_mem_result` | `np.ndarray` | Final memristive MVM result |
| `y_benchmark_result` | `np.ndarray` or `None` | CPU benchmark result |
| `error` | `np.ndarray` or `None` | Error vector |

---

## Methods

### `initializeMat(mat)`

Load and prepare the matrix for MVM.

```python
def initializeMat(self, mat: np.ndarray | None) -> None
```

Steps:

1. Calls `mca.initializeMatrix(mat)` (loads, scales).
2. Extracts dimensions and MCA grid parameters.
3. Computes `maxVRows` and `maxVCols`.
4. Calls `initializeVirtualizer()` if `virtualizationOn` is `True`.

---

### `initializeX(x)`

Prepare the input vector for tiled MVM.

```python
def initializeX(self, x: np.ndarray | None) -> None
```

Steps:

1. Reshapes `x` to `(origMatCols, 1)`.
2. Saves a copy as `globalX` and writes it to `<TMPDIR>/global_input_vec.txt`.
3. Applies min-max scaling (using `mca.scaleMatrix`).
4. Updates all per-tile `x` slices in `virtualizer`.

---

### `initializeVirtualizer()`

Decomposes `origMat` and `x` into tiles.

```python
def initializeVirtualizer(self) -> None
```

Creates entries `virtualizer[i, j]` for each tile `(i, j)`:

```python
{
    "rc_limits": [[row_start, row_end], [col_start, col_end]],
    "mat":       origMat[row_start:row_end, col_start:col_end],
    "x":         x[col_start:col_end, :]
}
```

Also creates `virtualizer[i]["y"]` — a zero buffer for accumulating column-tile results.

---

### `virtualParallelMatVec(i, j)`

Executes MVM for one tile `(i, j)`.

```python
def virtualParallelMatVec(self, i: int, j: int) -> None
```

Steps:

1. Broadcasts tile coordinates `[i, j]` to all workers via `comm.Bcast`.
2. Calls `mca.setMat(tile_mat)` to distribute the tile to workers.
3. Calls `mca.setX(tile_x)` to distribute the vector slice.
4. Calls `mca.parallelMatVec()` and accumulates result into `virtualizer[i]["y"]`.

---

### `parallelMatVec(correction=False)`

Executes the full distributed MVM, iterating over all tiles if virtualization is on.

```python
def parallelMatVec(self, correction: bool = False) -> None
```

After computing `y`:

- If `correction=True`: calls `addCorrectionY()` to reverse min-max scaling and stores in `y_mem_result`.
- If `correction=False`: stores raw scaled result in `y_mem_result`.

Results are saved to the following output files:

| File | Description |
|------|-------------|
| `<TMPDIR>/y_mem_result.txt` | Final MVM result (content depends on `correction` flag) |
| `<TMPDIR>/y_mem_result_reversal_applied.txt` | Result with min-max scaling reversal applied |
| `<TMPDIR>/y_mem_result_no_reversal.txt` | Result without scaling reversal (in `[0,1]` domain) |

---

### `addCorrectionY(n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum)`

Reverses the min-max scaling effects after MVM.

```python
def addCorrectionY(
    self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum
) -> np.ndarray
```

Formula for each row `i`:

```
Y[i] = y[i] * (a_max * x_max) + a_min * x_sum + x_min * a_row_sum[i] - n * a_min * x_min
```

---

### `removeCorrectionY(n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum)`

Inverts the `addCorrectionY` transformation.

```python
def removeCorrectionY(
    self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum
) -> np.ndarray
```

---

### `benchmarkMatVecParallel(hardwareOn=0, scalingOn=0, correction=False)`

Compares memristive MVM output against a NumPy CPU baseline.

```python
def benchmarkMatVecParallel(
    self,
    hardwareOn: int = 0,
    scalingOn: int = 0,
    correction: bool = False
) -> None
```

Reads the previously saved `y_mem_result.txt` and computes:

| Mode | CPU reference | Memristive reference |
|------|---------------|----------------------|
| `correction=True` | `A @ x` (original domain) | `y_mem_result` (after reversal) |
| `correction=False` | `A_scaled @ x_scaled` (scaled) | `y_mem_result` (scaled) |

Prints relative L2 and L∞ errors.

---

### `acquireMCAStats()`

Delegates to `mca.getMCAStats()`.

---

### `finalize()`

Broadcasts a sentinel `[-1, -1]` to signal all workers to exit their tile-wait loop.

```python
def finalize(self) -> None
```

---

## Module-Level Utilities

### `__out_path__(name)`

Returns the full path `<TMPDIR>/<name>`, creating the directory if needed. Falls back to the current directory if `TMPDIR` is not set.

### `__check_array_attributes__(array)`

Returns `(array_min, array_range, array_row_sum)`.

### `__minMax_Scale__(array)`

Returns `(scaled_array, min_val, range_val)`. Returns zeros if `range == 0`.

---

## See Also

- [`RootMCA`](../../src/core/RootMCA.md) — Lower-level coordinator used internally.
- [`MatVecSolver`](MatVecSolver.md) — High-level interface that creates `Root`.
- [`NonRoot`](NonRoot.md) — Worker counterpart.
