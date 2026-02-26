# API Reference

This section documents every public class and function in the MELISO source tree.

---

## `src.core` — MCA Process Roles

These classes implement the lower-level, MPI-aware logic for encoding matrices onto hardware and collecting results.

| Class | File | Role |
|-------|------|------|
| [`BaseMCA`](src/core/BaseMCA.md) | `src/core/BaseMCA.py` | Abstract base — reads config, validates grid |
| [`RootMCA`](src/core/RootMCA.md) | `src/core/RootMCA.py` | Root process — distributes matrix/vector, aggregates output |
| [`NonRootMCA`](src/core/NonRootMCA.md) | `src/core/NonRootMCA.py` | Worker process — encodes weights, runs local MVM |

---

## `src.cython` — Hardware Interface

Cython/C++ boundary layer that calls the NeuroSim simulator.

| Symbol | File | Description |
|--------|------|-------------|
| [`MelisoPy`](src/cython/MelisoPy.md) | `src/cython/mel.pyx` | Python-callable Cython extension wrapping `Meliso` C++ |
| [C++ Interface (`Meliso.pxd`)](src/cython/Meliso_interface.md) | `src/cython/Meliso.pxd` | Cython declaration of the C++ `meliso::Meliso` class |

---

## `solver.matvec` — High-Level MPI Solver

| Class | File | Description |
|-------|------|-------------|
| [`MatVecSolver`](solver/matvec/MatVecSolver.md) | `solver/matvec/MatVecSolver.py` | User-facing MVM interface |
| [`Root`](solver/matvec/Root.md) | `solver/matvec/Root.py` | Root-side coordinator with virtualization |
| [`NonRoot`](solver/matvec/NonRoot.md) | `solver/matvec/NonRoot.py` | Worker-side coordinator with virtualization |

---

## `solver.common` — Utilities

| Symbol | File | Description |
|--------|------|-------------|
| [`load_matrix_any`](solver/common/loaders.md) | `solver/common/loaders.py` | Auto-detect and load matrix files |
| [`make_toy_linear_systems`](solver/common/make_toy_linear_systems.md) | `solver/common/make_toy_linear_systems.py` | Generate test matrices |

---

## Driver Scripts

| Script | Description |
|--------|-------------|
| [`DistributedMatVec.py`](scripts/DistributedMatVec.md) | Main MVM entry-point |
| [`mlpInference.py`](scripts/mlpInference.md) | MLP inference on memristive hardware |
