# Matrix Loaders

**Module:** `solver.common.loaders`
**File:** `solver/common/loaders.py`

---

## Overview

This module provides a single public function, `load_matrix_any`, for loading matrices from various file formats with automatic format detection. It also handles optional row/column scaling via environment variables.

---

## Public API

### `load_matrix_any(path)`

Auto-detect the matrix format and return it as a dense NumPy array.

```python
def load_matrix_any(path: str) -> np.ndarray
```

**Parameters:**

| Parameter | Type | Description |
|-----------|------|-------------|
| `path` | `str` | Absolute or relative path to the matrix file |

**Returns:** `np.ndarray` with `dtype=float64`.

**Supported formats:**

| Extension | Loader | Notes |
|-----------|--------|-------|
| `.npz` | `numpy.load` | Sparse PDHG snapshot or dense array |
| `.mtx` | `scipy.io.mmread` | Matrix Market format |
| `.npy` | `numpy.load` | NumPy binary |

**Raises:**

- `ValueError` — invalid path, unsupported extension, or missing array keys.
- `RuntimeError` — `scipy` not installed when loading `.npz` (sparse) or `.mtx`.

---

#### `.npz` Loading Logic

The function inspects the keys inside the `.npz` file to distinguish two cases:

1. **Sparse PDHG snapshot** — keys contain `*_data`, `*_row`, `*_col`, `*_shape` suffixes.
   Delegates to `_load_constraint_from_npz()`.

2. **Dense array** — loads the array named `"A"` (or the first key if `"A"` is absent).

---

#### Environment Variables

| Variable | Effect on `.npz` loading |
|----------|--------------------------|
| `NPZ_PREFIX` | Array name prefix to search for in sparse NPZ files (default `"A"`) |
| `NPZ_TO_DENSE` | `"1"` (default) converts sparse COO to dense array; `"0"` keeps sparse |
| `APPLY_ROW_SCALING` | `"1"` multiplies each row `i` by `row_scale_vec[i]` |
| `APPLY_COL_SCALING` | `"1"` multiplies each column `j` by `col_scale_vec[j]` |

---

## Internal Functions

### `_load_constraint_from_npz(snapshot_path, prefix=None, to_dense=True)`

Load a sparse matrix stored in PDHG snapshot format from a `.npz` file.

```python
def _load_constraint_from_npz(
    snapshot_path: str,
    prefix: Optional[str] = None,
    to_dense: bool = True
) -> np.ndarray
```

**Expected keys in the `.npz`:**

```
<prefix>_data   — non-zero values
<prefix>_row    — row indices (COO format)
<prefix>_col    — column indices (COO format)
<prefix>_shape  — [rows, cols]
```

Tries prefixes `["A", "G", "H"]` in order if `prefix` is `None`.

Optionally applies row/column scaling via `_maybe_apply_scaling`.

---

### `_maybe_apply_scaling(A, z)`

Apply optional row and/or column scaling vectors stored inside the `.npz` archive.

```python
def _maybe_apply_scaling(A: np.ndarray, z: dict) -> np.ndarray
```

Controlled by `APPLY_ROW_SCALING` and `APPLY_COL_SCALING` environment variables.
Looks for `row_scale_vec` and `col_scale_vec` keys in the archive `z`.

---

## Examples

```python
from solver.common.loaders import load_matrix_any

# Load a Matrix Market file
A = load_matrix_any("inputs/matrices/bcsstk02.mtx")

# Load a NumPy binary
A = load_matrix_any("inputs/matrices/W1.npy")

# Load a sparse PDHG snapshot (searches for prefix "A" by default)
A = load_matrix_any("snapshots/problem_snapshot.npz")

# Apply row scaling stored in the archive
import os
os.environ["APPLY_ROW_SCALING"] = "1"
A = load_matrix_any("snapshots/problem_snapshot.npz")
```

---

## See Also

- [`make_toy_linear_systems`](make_toy_linear_systems.md) — Generate test matrices.
- [`RootMCA.readMatrix`](../../src/core/RootMCA.md#readmatrixfilename) — The simpler matrix reader used by `RootMCA`.
