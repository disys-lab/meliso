# Toy Linear System Generator

**Module:** `solver.common.make_toy_linear_systems`
**File:** `solver/common/make_toy_linear_systems.py`

---

## Overview

This script generates three small toy linear systems (`Ax = b`) and saves each case as a set of `.npy` files. It is intended for quick testing and validation of the MELISO MVM pipeline without requiring external matrix data.

---

## Functions

### `save_case(folder, name, A, b, xstar)`

Save a single test case to disk.

```python
def save_case(
    folder: str,
    name: str,
    A: np.ndarray,
    b: np.ndarray,
    xstar: np.ndarray
) -> None
```

Creates the directory `folder` if it does not exist and writes three files:

| File | Contents |
|------|----------|
| `<folder>/<name>_A.npy` | Coefficient matrix |
| `<folder>/<name>_b.npy` | Right-hand side vector |
| `<folder>/<name>_xstar.npy` | True solution vector |

---

### `main()`

Generate and save three toy linear systems to the `toy_cases/` directory.

```python
def main() -> None
```

#### Test Cases

| Case | Matrix shape | Type | Description |
|------|-------------|------|-------------|
| `case1` | `2×2` | Square | `A = [[2,-1],[-3,4]]`, `x* = [1.5,-2.0]` |
| `case2` | `3×3` | Square | `A = [[1,-2,0.5],[0,3,-1],[-4,1,2]]`, `x* = [-1,0.5,2]` |
| `case3` | `5×2` | Overdetermined | `A = [[1,2],[-2,1],[0.5,-1],[3,-0.5],[-1.5,-2]]`, `x* = [0.75,-1.25]` |

For each case, prints:

- The true solution `x*`.
- The NumPy-computed solution (via `np.linalg.solve` for square, `np.linalg.lstsq` for overdetermined).
- The `||x_numpy - x*||` residual.

---

## Usage

```bash
python3 solver/common/make_toy_linear_systems.py
```

This creates:

```
toy_cases/
├── case1_A.npy
├── case1_b.npy
├── case1_xstar.npy
├── case2_A.npy
├── case2_b.npy
├── case2_xstar.npy
├── case3_A.npy
├── case3_b.npy
└── case3_xstar.npy
```

---

## See Also

- [`load_matrix_any`](loaders.md) — Load the saved `.npy` files.
