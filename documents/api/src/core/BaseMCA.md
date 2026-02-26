# `BaseMCA`

**Module:** `src.core.BaseMCA`
**File:** `src/core/BaseMCA.py`

---

## Overview

`BaseMCA` is the abstract base class for all MCA (Memristor Crossbar Array) process roles in MELISO.
It handles:

- MPI communicator initialization.
- Reading and validating the experiment configuration YAML (`EXP_CONFIG_FILE`).
- Extracting the MCA grid dimensions (`mca_rows`, `mca_cols`) and cell dimensions (`cell_rows`, `cell_cols`).
- Validating that the number of MPI processes matches the MCA grid size.

Both [`RootMCA`](RootMCA.md) and [`NonRootMCA`](NonRootMCA.md) inherit from this class.

---

## Device Type Reference

The first argument to the `Meliso` C++ constructor sets the device model:

| Code | Name |
|------|------|
| `0` | `IdealDevice` |
| `1` | `RealDevice` (default) |
| `2` | `MeasuredDevice` |
| `3` | `SRAM` |
| `4` | `DigitalNVM` |
| `5` | `HybridCell` |
| `6` | `_2T1F` |

---

## Class Definition

```python
class BaseMCA:
    def __init__(self, comm): ...
```

### Parameters

| Parameter | Type | Description |
|-----------|------|-------------|
| `comm` | `mpi4py.MPI.Comm` | MPI communicator (typically `MPI.COMM_WORLD`) |

---

## Constructor Behaviour

1. Extracts `rank` and `size` from `comm`.
2. Reads the path to the experiment config from the `EXP_CONFIG_FILE` environment variable.
   Exits with an error message if the variable is not set.
3. Sets `ROOT_PROCESS_RANK = size - 1` (the last rank is always the root).
4. Calls `readExpConfig()` to parse the YAML and populate grid/cell dimensions.
5. Validates `size - 1 == mca_rows * mca_cols`.

---

## Attributes

| Attribute | Type | Description |
|-----------|------|-------------|
| `comm` | `MPI.Comm` | MPI communicator |
| `rank` | `int` | This process's MPI rank |
| `size` | `int` | Total number of MPI processes |
| `ROOT_PROCESS_RANK` | `int` | Rank of the root process (`size - 1`) |
| `expConfigFile` | `str` | Path to the experiment YAML config file |
| `exp_config` | `dict` | Parsed experiment configuration dictionary |
| `mcaRows` | `int` | Number of MCA rows in the hardware grid |
| `mcaCols` | `int` | Number of MCA columns in the hardware grid |
| `cellRows` | `int` | Number of cell rows per MCA unit |
| `cellCols` | `int` | Number of cell columns per MCA unit |
| `decomposition_dir` | `str` | Directory for temporary matrix chunk files (default `/tmp/`) |
| `distributed` | `int` | `1` if distributed mode is active, `0` otherwise |
| `matrix_name` | `str` | Name of the matrix (from `exp_params.matrix_name`) |
| `ERR_CORR` | `int` | Error correction flag (`1` = enabled). Can be overridden by `EC` env var |
| `num_mca_stats` | `int` | Number of MCA statistics collected (always `8`) |
| `useMPI4MatDist` | `bool` | Use MPI `Send`/`Recv` for matrix distribution (default `True`) |

---

## Methods

### `readExpConfig(expConfigFile)`

Parses the experiment YAML file and populates grid and cell dimensions.

```python
def readExpConfig(self, expConfigFile: str) -> None
```

**Raises:** `Exception` if the file cannot be opened, or if required keys (`mca_rows`, `mca_cols`) are missing.

**Config keys read:**

| YAML path | Attribute set |
|-----------|--------------|
| `device_config.cell_rows` | `self.cellRows` |
| `device_config.cell_cols` | `self.cellCols` |
| `exp_params.distributed.decomposition_dir` | `self.decomposition_dir` |
| `exp_params.distributed.mca_rows` | `self.mcaRows` |
| `exp_params.distributed.mca_cols` | `self.mcaCols` |

The `TMPDIR` environment variable overrides `decomposition_dir` if set.

---

### `getDecompositionDir()`

Returns the path to the directory where matrix chunk files (`.npy`) are stored.

```python
def getDecompositionDir(self) -> str
```

The directory name encodes the matrix name and grid dimensions:

```
<decomposition_dir>/<matrix_name>-<mcaRows>_<mcaCols>_<cellRows>_<cellCols>/
```

---

### `getMCAStats()`

No-op placeholder. Overridden in [`RootMCA`](RootMCA.md) and [`NonRootMCA`](NonRootMCA.md).

---

## Experiment Configuration YAML Format

```yaml
exp_params:
  turnOnHardware: 1
  turnOnScaling: 0
  matrix_name: "bcsstk02"
  matrix_file: "inputs/matrices/bcsstk02.mtx"
  errCorr: 1
  interpolants: 3
  distributed:
    decomposition_dir: "/tmp/"
    mca_rows: 1
    mca_cols: 1

device_config:
  root: "config_files/"
  cell_rows: 66
  cell_cols: 66
  assignment:
    device_TaOx-HfOx.yaml: [[-1]]
```

---

## Environment Variables

| Variable | Effect |
|----------|--------|
| `EXP_CONFIG_FILE` | **Required.** Path to the experiment YAML. |
| `EC` | Override the `errCorr` flag from YAML (`0` or `1`). |
| `TMPDIR` | Override `decomposition_dir` for chunk files. |

---

## See Also

- [`RootMCA`](RootMCA.md) — Subclass for the root/coordinator process.
- [`NonRootMCA`](NonRootMCA.md) — Subclass for worker processes.
