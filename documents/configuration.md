# Configuration Reference

MELISO+ uses two types of YAML configuration files:

1. **Experiment config** (`exp*.yaml`) — defines the matrix, MCA grid layout, and which device configs to use.
2. **Device config** (`device_*.yaml`) — defines the physical parameters of a memristor device material.

Both files must exist and be correctly referenced before running any experiment.

---

## Experiment Configuration (`exp*.yaml`)

### Full Schema

```yaml
exp_params:
  turnOnHardware: <int>        # required
  turnOnScaling: <int>         # required
  matrix_name: <string>        # required
  matrix_file: <string>        # required
  errCorr: <int>               # optional
  interpolants: <int>          # optional
  distributed:
    decomposition_dir: <string> # optional
    mca_rows: <int>             # required
    mca_cols: <int>             # required

device_config:
  root: <string>               # required
  cell_rows: <int>             # required
  cell_cols: <int>             # required
  assignment:
    <device_file.yaml>: <rank_list>  # required (one or more entries)
```

---

### `exp_params` Fields

#### `turnOnHardware`
**Type:** `int` — `0` or `1`
**Required:** Yes

Controls whether the NeuroSim hardware simulator is active.

| Value | Behaviour |
|-------|-----------|
| `1` | Hardware simulation is on. Weights are encoded with device noise and non-idealities. |
| `0` | Hardware simulation is off. The MVM is effectively a simple floating-point multiply — useful for debugging the pipeline. |

---

#### `turnOnScaling`
**Type:** `int` — `0` or `1`
**Required:** Yes

Controls whether an additional internal conductance scaling is applied inside the C++ backend. In most experiments this is set to `0`.

| Value | Behaviour |
|-------|-----------|
| `1` | Enable internal scaling inside NeuroSim. |
| `0` | Disable internal scaling (recommended). |

---

#### `matrix_name`
**Type:** `string`
**Required:** Yes

A short identifier for the matrix. Used to name the temporary matrix chunk directory:

```
<decomposition_dir>/<matrix_name>-<mca_rows>_<mca_cols>_<cell_rows>_<cell_cols>/
```

Does not need to match the filename — it is only used for naming purposes.

---

#### `matrix_file`
**Type:** `string` (file path, relative to the project root)
**Required:** Yes

Path to the input matrix file. Supported formats:

| Extension | Format |
|-----------|--------|
| `.mtx` | Matrix Market (sparse or dense) |
| `.npy` | NumPy binary array |
| `.csv` | Comma-separated values |
| `.txt` | Comma-separated values |
| `.npz` | NumPy archive (sparse PDHG snapshot or dense) |

---

#### `errCorr`
**Type:** `int` — `0` or `1`
**Required:** No (default: `1`)

Enable or disable the multi-step error correction algorithm. Can also be overridden at runtime with the `EC` environment variable.

| Value | Behaviour |
|-------|-----------|
| `1` | Error correction on. More accurate but slower (requires multiple MVM passes per tile). |
| `0` | Error correction off. Faster but less accurate, especially for noisy devices. |

---

#### `interpolants`
**Type:** `int`
**Required:** No (default: `3`)

Number of interpolation points used by the NeuroSim device model for piecewise-linear conductance curves. Higher values improve accuracy but increase simulation time. Typical values are `3` to `5`.

---

#### `distributed.decomposition_dir`
**Type:** `string` (directory path)
**Required:** No (default: `/tmp/` or `TMPDIR` env var)

Directory where matrix sub-block files (`.npy`) are written during decomposition. Must be writable and accessible by all MPI processes. On shared-filesystem clusters, use a path on the shared filesystem rather than `/tmp/`.

Can be overridden at runtime with the `TMPDIR` environment variable (takes priority over YAML).

---

#### `distributed.mca_rows`
**Type:** `int`
**Required:** Yes (if `distributed` block is present)

Number of rows in the MCA grid. Together with `mca_cols`, determines how the matrix is partitioned and how many MPI worker processes are needed.

**Constraint:** `mpi_processes = mca_rows × mca_cols + 1`

---

#### `distributed.mca_cols`
**Type:** `int`
**Required:** Yes (if `distributed` block is present)

Number of columns in the MCA grid.

---

### `device_config` Fields

#### `device_config.root`
**Type:** `string` (directory path, relative to project root)
**Required:** Yes

The directory prefix prepended to device config filenames in the `assignment` map.

Example: if `root: "config_files/"` and `assignment` contains `device_EpiRAM.yaml`, the full path resolved is `config_files/device_EpiRAM.yaml`.

---

#### `device_config.cell_rows`
**Type:** `int`
**Required:** Yes

Number of cell rows in each individual MCA chip. Combined with `mca_rows`, sets the total row capacity of the hardware grid:

```
total_row_capacity = mca_rows × cell_rows
```

The input matrix is zero-padded to this capacity if needed.

---

#### `device_config.cell_cols`
**Type:** `int`
**Required:** Yes

Number of cell columns in each individual MCA chip.

```
total_col_capacity = mca_cols × cell_cols
```

---

#### `device_config.assignment`
**Type:** `dict`
**Required:** Yes

Maps device config filenames to the MCA ranks that should use them. Enables heterogeneous systems (different device materials for different chips).

**Format:** `{ <device_file.yaml>: <rank_list> }`

**Rank list syntax:**

| Value | Meaning |
|-------|---------|
| `[[-1]]` | All MCA ranks use this device config (homogeneous system) |
| `[[0, 3]]` | Ranks 0, 1, 2 use this device config (range, exclusive end) |
| `[[0]]` | Only rank 0 uses this device config |

**Homogeneous example:**
```yaml
assignment:
  device_EpiRAM.yaml: [[-1]]
```

**Heterogeneous example (2×2 grid = 4 ranks):**
```yaml
assignment:
  device_Ag-aSi.yaml:    [[0]]
  device_AlOx-HfO2.yaml: [[1]]
  device_EpiRAM.yaml:    [[2]]
  device_TaOx-HfOx.yaml: [[3]]
```

---

### Complete Experiment Config Examples

**Minimal — 1 array, EpiRAM, 66×66 cells:**
```yaml
exp_params:
  turnOnHardware: 1
  turnOnScaling: 0
  matrix_name: "bcsstk02"
  matrix_file: "inputs/matrices/bcsstk02.mtx"
  distributed:
    decomposition_dir: "/tmp/"
    mca_rows: 1
    mca_cols: 1

device_config:
  root: "config_files/"
  cell_rows: 66
  cell_cols: 66
  assignment:
    device_EpiRAM.yaml: [[-1]]
```
Launch with: `mpiexec -n 2 python3 DistributedMatVec.py`

---

**3×3 grid, Ag-aSi, 512×512 cells:**
```yaml
exp_params:
  turnOnHardware: 1
  turnOnScaling: 0
  matrix_name: "add32"
  matrix_file: "inputs/matrices/add32.mtx"
  distributed:
    decomposition_dir: "/tmp/"
    mca_rows: 3
    mca_cols: 3

device_config:
  root: "config_files/"
  cell_rows: 512
  cell_cols: 512
  assignment:
    device_Ag-aSi.yaml: [[-1]]
```
Launch with: `mpiexec -n 10 python3 DistributedMatVec.py`

---

## Device Configuration (`device_*.yaml`)

### Full Schema

```yaml
Device:
  ConductanceProperties:
    maxConductance: <float>         # required
    minConductance: <float>         # required
    avgMaxConductance: <float|alias> # required
    avgMinConductance: <float|alias> # required
    conductance: <float|alias>       # required
    conductancePrev: <float|alias>   # required

  ReadProperties:
    readVoltage: <float>
    readPulseWidth: <float>

  WriteProperties:
    writeVoltageLTP: <float>        # required
    writeVoltageLTD: <float>        # required
    writePulseWidthLTP: <float>     # required
    writePulseWidthLTD: <float>     # required
    writeEnergy: <float>
    maxNumLevelLTP: <int>           # required
    maxNumLevelLTD: <int>           # required
    numPulse: <int>

  DeviceConfiguration:
    cmosAccess: <bool>
    FeFET: <bool>
    gateCapFeFET: <float>
    resistanceAccess: <float>
    nonlinearIV: <bool>
    nonIdenticalPulse: <bool>
    NL: <float>

  DeviceVariation:
    NL_LTP: <float>                 # required
    NL_LTD: <float>                 # required
    sigmaDtoD: <float>              # required
    sigmaCtoC: <float>              # required

  ReadNoise:
    enabled: <bool>
    sigmaReadNoise: <float>

  ConductanceRangeVariation:
    enabled: <bool>
    maxConductanceVar: <float>
    minConductanceVar: <float>

  PhysicalDimensions:
    heightInFeatureSize: <int>
    widthInFeatureSize: <int>

  DeviceMaterials:
    material: <string>

  NonlinearWrite:
    enabled: <bool>

  NonIdenticalPulseSettings:
    VinitLTP: <float>
    VstepLTP: <float>
    VinitLTD: <float>
    VstepLTD: <float>
    PWinitLTP: <float>
    PWstepLTP: <float>
    PWinitLTD: <float>
    PWstepLTD: <float>
    writeVoltageSquareSum: <float>
```

---

### Field Reference

#### `ConductanceProperties`

| Field | Unit | Description |
|-------|------|-------------|
| `maxConductance` | S (siemens) | Maximum achievable cell conductance |
| `minConductance` | S | Minimum achievable cell conductance (reset state) |
| `avgMaxConductance` | S | Average max conductance; may reference `maxConductance` by alias |
| `avgMinConductance` | S | Average min conductance; may reference `minConductance` by alias |
| `conductance` | S | Initial conductance state (typically `minConductance`) |
| `conductancePrev` | S | Previous conductance state (used in write algorithm) |

**Alias values:** Fields that accept `"maxConductance"` or `"minConductance"` as string values will resolve to the corresponding numeric values at runtime.

---

#### `WriteProperties`

| Field | Unit | Description |
|-------|------|-------------|
| `writeVoltageLTP` | V | Voltage pulse for long-term potentiation (weight increase) |
| `writeVoltageLTD` | V | Voltage pulse for long-term depression (weight decrease) |
| `writePulseWidthLTP` | s | Pulse duration for LTP |
| `writePulseWidthLTD` | s | Pulse duration for LTD |
| `maxNumLevelLTP` | — | Maximum discrete conductance levels for LTP |
| `maxNumLevelLTD` | — | Maximum discrete conductance levels for LTD |
| `writeEnergy` | J | Energy per write pulse (0 = compute from model) |
| `numPulse` | — | Number of pulses per write step (0 = determined by model) |

---

#### `DeviceVariation`

These four parameters control the stochastic non-idealities of the device model. They are the most impactful parameters on MVM accuracy.

| Field | Description | Effect on accuracy |
|-------|-------------|-------------------|
| `NL_LTP` | Non-linearity exponent for LTP | Higher absolute value → more distortion during weight increase |
| `NL_LTD` | Non-linearity exponent for LTD | Higher absolute value → more distortion during weight decrease |
| `sigmaDtoD` | Device-to-device variation (σ) | Spread between chips; `0` = no D2D variation |
| `sigmaCtoC` | Cycle-to-cycle variation (σ) | Noise per write operation; higher → less repeatable writes |

Setting all four to `0` produces a deterministic, linear device (closest to `IdealDevice`).

---

#### `ReadProperties`

| Field | Unit | Description |
|-------|------|-------------|
| `readVoltage` | V | Voltage applied during MVM (read operation) |
| `readPulseWidth` | s | Duration of each read pulse |

---

## Device Materials

MELISO+ ships with four pre-characterised device materials. Choose based on your target application and acceptable noise levels.

### Comparison Table

| Material | File | `maxG` (S) | `minG` (S) | LTP levels | LTD levels | `NL_LTP` | `NL_LTD` | `sigmaCtoC` |
|----------|------|-----------|-----------|-----------|-----------|---------|---------|-----------|
| **EpiRAM** (Ag:SiGe) | `device_EpiRAM.yaml` | 1.23e-5 | 2.46e-7 | 256 | 256 | 0.5 | -0.5 | 0.02 |
| **Ag-aSi** (Ag:a-Si) | `device_Ag-aSi.yaml` | 3.85e-8 | 3.08e-9 | 97 | 100 | 2.4 | -4.88 | 0.035 |
| **AlOx-HfO2** | `device_AlOx-HfO2.yaml` | 5.92e-5 | 1.34e-5 | 40 | 40 | 1.94 | -0.61 | 0.05 |
| **TaOx-HfOx** | `device_TaOx-HfOx.yaml` | 1.0e-5 | 1.0e-6 | 128 | 128 | 0.04 | -0.63 | 0.037 |

### Guidance

| Goal | Recommended material |
|------|---------------------|
| Highest precision / lowest noise | **EpiRAM** — lowest non-linearity, lowest cycle-to-cycle variation |
| Most discrete levels | **EpiRAM** — 256 levels per direction |
| Closest to linear write behaviour | **TaOx-HfOx** — `NL_LTP` of only 0.04 |
| Fastest write pulses | **TaOx-HfOx** — 50 ns pulses vs. 300 µs for Ag-aSi |
| Highest conductance range ratio | **Ag-aSi** — ~12.5× ratio (`maxG/minG`) |

> Note: All four materials use CMOS access transistors (`cmosAccess: true`) and have nonlinear IV disabled by default.

---

## MCA Grid Sizing Rules

When choosing `mca_rows`, `mca_cols`, `cell_rows`, and `cell_cols`, these constraints must hold:

```
mca_rows × cell_rows  ≥  matrix_rows
mca_cols × cell_cols  ≥  matrix_cols
mpi_processes = mca_rows × mca_cols + 1
```

If the matrix is smaller than the grid capacity, MELISO+ will zero-pad it automatically. If the matrix is larger, virtualization (tiling) is applied automatically — the matrix is processed in sequential passes without any changes to the config.

**Example sizing for a 500×500 matrix:**

| `mca_rows` | `mca_cols` | `cell_rows` | `cell_cols` | MPI procs | Notes |
|-----------|-----------|------------|------------|-----------|-------|
| 1 | 1 | 512 | 512 | 2 | Single chip, matrix fits with padding |
| 2 | 2 | 256 | 256 | 5 | 4 chips, matrix fits exactly |
| 1 | 1 | 64 | 64 | 2 | Single chip, virtualization handles tiling |
