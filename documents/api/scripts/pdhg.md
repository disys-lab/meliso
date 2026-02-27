# Driver Script: `pdhg.py`

**File:** `pdhg.py`
**Authors:** Huynh Quang Nguyen Vo (Oklahoma State University)

---

## Overview

`pdhg.py` implements a **Primal-Dual Hybrid Gradient (PDHG)** solver for linear programs, accelerated by the MELISO+ framework for distributed matrix-vector multiplication (MVM) on memristive crossbar hardware.

The script solves problems of the form:

```
minimize    c^T x
subject to  Ax = b,  x >= 0
```

using the iterative PDHG algorithm with extrapolation. Each PDHG iteration requires two matrix-vector products (`A*x_bar` and `A^T*mu`), both offloaded to MELISO+ hardware.

The script must be launched via `mpiexec` with the same number of processes as required by the MCA configuration:

```bash
A_FILE=inputs/A.csv B_FILE=inputs/b.csv C_FILE=inputs/c.csv \
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
  mpiexec -n 2 python3 pdhg.py
```

---

## Class: `PDHGSolver`

```python
class PDHGSolver:
    RESULT_FILENAME    = "y_mem_result.csv"
    X_ITERATES_FILENAME = "x_iterates.csv"
    LOG_FILENAME       = "x_log.txt"
```

### Class Constants

| Constant | Value | Description |
|----------|-------|-------------|
| `RESULT_FILENAME` | `"y_mem_result.csv"` | Output file for MVM results |
| `X_ITERATES_FILENAME` | `"x_iterates.csv"` | Output file for all primal iterates |
| `LOG_FILENAME` | `"x_log.txt"` | Convergence log file |

---

### `__init__(A, b, c, num_iterations)`

Initialize the PDHG solver with problem data and iteration count.

```python
def __init__(
    self,
    A: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
    num_iterations: int
) -> None
```

| Parameter | Type | Description |
|-----------|------|-------------|
| `A` | `np.ndarray` | Constraint matrix of shape `(m, n)` |
| `b` | `np.ndarray` | Right-hand side vector of shape `(m,)` |
| `c` | `np.ndarray` | Objective cost vector of shape `(n,)` |
| `num_iterations` | `int` | Maximum number of PDHG iterations |

Internally creates a `MatVecSolver` instance stored in `self.mv_solver`. This triggers MPI initialization for all worker processes.

---

### Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `A` | `np.ndarray` | Constraint matrix |
| `b` | `np.ndarray` | RHS vector |
| `c` | `np.ndarray` | Cost vector |
| `n_primal` | `int` | Primal dimension (`A.shape[1]`) |
| `n_dual` | `int` | Dual dimension (`A.shape[0]`) |
| `tol` | `float` | Convergence tolerance (`1e-6`) |
| `theta` | `float` | Extrapolation parameter (`1.0`) |
| `num_iterations` | `int` | Maximum number of iterations |
| `x_iterates` | `List[np.ndarray]` | All primal iterates collected during `solve()` |
| `x_bar` | `np.ndarray` or `None` | Extrapolated primal variable (updated each iteration) |
| `x_avg` | `np.ndarray` or `None` | Ergodic (averaged) solution after `solve()` |
| `mv_solver` | `MatVecSolver` | MELISO+ interface for accelerated MVM |

---

### `_compute_matvec(matrix, vector)`

Compute a single matrix-vector product using the MELISO+ accelerator.

```python
def _compute_matvec(
    self,
    matrix: np.ndarray,
    vector: np.ndarray
) -> np.ndarray
```

| Parameter | Description |
|-----------|-------------|
| `matrix` | Matrix to multiply (e.g. `A` or `A.T`) |
| `vector` | Input vector |

Steps:

1. Calls `mv_solver.solverObject.initializeMat(matrix)` to load the matrix on the root process.
2. Calls `mv_solver.solverObject.initializeX(vector)` to load the vector on the root process.
3. Calls `mv_solver.matVec(correction=True)` — applies min-max scaling reversal so results are in the original domain.
4. Calls `mv_solver.finalize()` to signal workers after the MVM.
5. Calls `mv_solver.acquireMCAStats()` to gather hardware statistics.
6. Returns the result via `mv_solver.acquireResults()`.

!!! note
    `correction=True` is used so that the result is returned in the original mathematical domain (not the normalised `[0,1]` domain), which is required for correct PDHG updates.

---

### `_project_dual(mu)` (static)

Project dual variables onto the non-negative orthant.

```python
@staticmethod
def _project_dual(mu: np.ndarray) -> np.ndarray
```

Returns `np.maximum(mu, 0)`.

---

### `_project_primal(x)` (static)

Project primal variables onto the non-negative orthant.

```python
@staticmethod
def _project_primal(x: np.ndarray) -> np.ndarray
```

Returns `np.maximum(x, 0)`.

---

### `_compute_stepsize(A)` (static)

Compute the primal and dual step sizes from the spectral norm of `A`.

```python
@staticmethod
def _compute_stepsize(A: np.ndarray) -> tuple[float, float]
```

```
maximum_step = 1 / ||A||_2
primal_step  = maximum_step
dual_step    = maximum_step
```

Returns `(primal_step, dual_step)`. Using equal primal and dual steps equal to `1/||A||_2` ensures convergence of the PDHG algorithm.

---

### `solve()`

Execute the PDHG iterations and return the final and averaged solutions.

```python
def solve(self) -> tuple[np.ndarray, np.ndarray]
```

**Returns:** `(x_bar, x_avg)` — the extrapolated final iterate and the ergodic average.

#### Algorithm

```
Initialize: x = 0,  mu = 0,  x_bar = 0
Compute: primal_step, dual_step = 1 / ||A||_2

For k = 0, 1, ..., num_iterations-1:

    # Dual update
    mv_result = A * x_bar              (via MELISO+)
    mu_tilde  = mu + dual_step * (mv_result - b)
    mu_next   = max(mu_tilde, 0)        (project onto R+)

    # Primal update
    mv_result = A^T * mu_next           (via MELISO+)
    x_grad    = x - primal_step * (mv_result + c)
    x_next    = max(x_grad, 0)          (project onto R+)

    # Extrapolation
    x_bar = x_next + theta * (x_next - x)

    # Check convergence
    if ||x_next - x||_2 < tol:
        break

    x, mu = x_next, mu_next
    Append x_next to x_log.txt

Save all iterates to x_iterates.csv
x_avg = mean(x_iterates)
```

#### Output Files

| File | Contents |
|------|----------|
| `x_log.txt` | Primal iterate appended each iteration; removed and recreated at the start of each `solve()` call |
| `x_iterates.csv` | All collected primal iterates as a CSV matrix; removed and recreated at the end of `solve()` |

---

## Module-Level Function: `main()`

MPI-aware entry point that separates root-process algorithm execution from worker-process loops.

```python
def main() -> None
```

### Root Process (`rank == size - 1`)

1. Reads `A`, `b`, `c` from files at paths given by environment variables `A_FILE`, `B_FILE`, `C_FILE`.
2. Creates a `PDHGSolver` with `num_iterations=100000`.
3. Calls `solver.solve()` and prints the optimal solution and objective value.
4. Broadcasts a termination signal to worker processes.

### Worker Processes (`rank < size - 1`)

Loop waiting for broadcasts from the root process. Each iteration of the PDHG loop triggers two `MatVecSolver().matVec(correction=True)` calls (one for `A*x_bar`, one for `A^T*mu`). Worker processes exit when they receive `True` from the root's termination broadcast.

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `EXP_CONFIG_FILE` | **Required.** Path to the experiment YAML configuration file |
| `A_FILE` | **Required.** Path to the constraint matrix CSV file |
| `B_FILE` | **Required.** Path to the RHS vector CSV file |
| `C_FILE` | **Required.** Path to the cost vector CSV file |
| `DT` | Device type (0–6); defaults to `1` (RealDevice) |
| `ITER_LIMIT` | Write-and-verify iteration cap |
| `TMPDIR` | Temporary directory for intermediate MVM output files |
| `REPORT_PATH` | Path for MCA statistics report |

---

## PDHG Iteration Data Flow

```
Root Process                            Worker Processes
     │                                        │
     │  solve()                               │
     │   ┌───────────────────────────────┐    │
     │   │ Dual update:                  │    │
     │   │  _compute_matvec(A, x_bar)    │    │
     │   │   → matVec(correction=True)   │◄───┤ tile MVM
     │   │   → finalize()               │────►│ exit loop
     │   │   → acquireResults()         │    │
     │   │                               │    │
     │   │ Primal update:                │    │
     │   │  _compute_matvec(A.T, mu_next)│    │
     │   │   → matVec(correction=True)   │◄───┤ tile MVM
     │   │   → finalize()               │────►│ exit loop
     │   └───────────────────────────────┘    │
     │  ... repeat for num_iterations ...     │
     │                                        │
     │  bcast(True) ─────────────────────────►│ exit
```

---

## Example Usage

```bash
# Generate toy problem data first
python3 -c "
import numpy as np
A = np.random.rand(4, 8)
b = np.random.rand(4)
c = np.random.rand(8)
np.savetxt('inputs/A.csv', A, delimiter=',')
np.savetxt('inputs/b.csv', b, delimiter=',')
np.savetxt('inputs/c.csv', c, delimiter=',')
"

# Run PDHG with 1x1 MCA grid (2 MPI processes)
A_FILE=inputs/A.csv \
B_FILE=inputs/b.csv \
C_FILE=inputs/c.csv \
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
DT=1 ITER_LIMIT=21 \
  mpiexec -n 2 python3 pdhg.py
```

---

## See Also

- [`MatVecSolver`](../solver/matvec/MatVecSolver.md) — The MVM interface used for accelerated computation.
- [`DistributedMatVec.py`](DistributedMatVec.md) — Simpler single-MVM driver script.
- [`mlpInference.py`](mlpInference.md) — MLP inference driver using the same framework.
- [Tutorials](../../tutorials/run_distributed_MVM.md) — Step-by-step guide for running MELISO+.
