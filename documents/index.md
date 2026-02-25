# MELISO — Memristor Linear Solver for Optimization

**MELISO** (also referred to as **MELISO+**) is a Python/Cython framework for performing distributed matrix-vector multiplication (MVM) on memristive crossbar array (MCA) hardware, backed by the [NeuroSim](https://github.com/neurosim) C++ simulator.

---

## Overview

MELISO provides a high-level Python interface that:

- Distributes a matrix across an MPI process pool, each of which maps to a physical (or simulated) memristor crossbar array.
- Encodes matrix weights onto hardware conductances using an iterative write-and-verify scheme.
- Optionally applies a multi-step error-correction algorithm to mitigate device non-idealities.
- Supports **virtualization** — tiling matrices that are larger than the hardware capacity across multiple sequential passes.
- Exposes hardware performance statistics (area, latency, energy) from the NeuroSim backend.

---

## Architecture

```
User Script  (DistributedMatVec.py / mlpInference.py)
     │
     ▼
MatVecSolver  (solver/matvec/MatVecSolver.py)
     ├── Root  (solver/matvec/Root.py)          ← MPI rank = size-1
     │      └── RootMCA  (src/core/RootMCA.py)
     └── NonRoot  (solver/matvec/NonRoot.py)    ← MPI ranks 0 … size-2
            └── NonRootMCA  (src/core/NonRootMCA.py)
                       └── MelisoPy  (src/cython/mel.pyx)
                                  └── Meliso C++ class  (src/cython/Meliso.cpp)
```

---

## Quick-Start

```bash
# Clone and build
git clone https://github.com/disys-lab/meliso.git
cd meliso
mkdir build
export PYTHONPATH="${PYTHONPATH}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:./build/"
make all

# Run a 1-array experiment (1 worker + 1 root = 2 MPI ranks)
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
  mpiexec -n 2 python3 DistributedMatVec.py
```

---

## Package Layout

| Path | Description |
|------|-------------|
| `src/core/` | Python base and process-role classes (`BaseMCA`, `RootMCA`, `NonRootMCA`) |
| `src/cython/` | Cython wrapper (`MelisoPy`) and C++ declarations (`Meliso.pxd`) |
| `solver/matvec/` | High-level MPI solver interface (`MatVecSolver`, `Root`, `NonRoot`) |
| `solver/common/` | Matrix loaders and toy-problem generators |
| `DistributedMatVec.py` | Main entry-point driver script |
| `mlpInference.py` | MLP neural-network inference on memristive hardware |
| `config_files/` | Experiment and device YAML configuration files |
| `inputs/` | Input matrices and vectors |
| `documents/` | This documentation |

---

## Documentation Sections

- [API Reference](api/index.md) — All classes and public functions
    - [`src.core`](api/src/core/BaseMCA.md) — Core MCA process roles
    - [`src.cython`](api/src/cython/MelisoPy.md) — Cython/C++ hardware interface
    - [`solver.matvec`](api/solver/matvec/MatVecSolver.md) — High-level solver
    - [`solver.common`](api/solver/common/loaders.md) — Utilities
    - [Driver scripts](api/scripts/DistributedMatVec.md) — Entry-point scripts
- [Tutorials](tutorials/run_distributed_MVM.md) — Step-by-step guides

---

## Supported Device Types

| Code | Name | Notes |
|------|------|-------|
| `0` | `IdealDevice` | No non-idealities |
| `1` | `RealDevice` | Default — resistive RAM with write noise |
| `2` | `MeasuredDevice` | Experimentally measured I-V curves |
| `3` | `SRAM` | Static RAM cell model |
| `4` | `DigitalNVM` | Multi-level NVM (digital) |
| `5` | `HybridCell` | Hybrid analog/digital cell |
| `6` | `_2T1F` | 2-transistor 1-ferroelectric capacitor |

---

## Environment Variables

| Variable | Purpose | Default |
|----------|---------|---------|
| `EXP_CONFIG_FILE` | Path to experiment YAML | **required** |
| `DT` | Device type (0–6) | `1` |
| `ITER_LIMIT` | Write-and-verify iteration cap | `21` |
| `OVERRIDE` | Force full iteration count (`0`/`1`) | `0` |
| `EC` | Error correction flag (`0`/`1`) | from YAML |
| `XVEC_PATH` | Input vector file | `inputs/vectors/input_x.txt` |
| `REPORT_PATH` | Output report file | `default_report.txt` |
| `TMPDIR` | Temp directory for matrix chunks | `/tmp/` |
| `MELISO_SRC_PATH` | Override source root path | `../../` |
| `NPZ_PREFIX` | Array name inside `.npz` | `A` |
| `NPZ_TO_DENSE` | Convert sparse NPZ to dense | `1` |
| `APPLY_ROW_SCALING` | Apply row-scaling vectors | `0` |
| `APPLY_COL_SCALING` | Apply column-scaling vectors | `0` |
