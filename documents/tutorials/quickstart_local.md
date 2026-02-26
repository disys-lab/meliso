# Quick-Start: Running Locally

This tutorial walks you through running your first MELISO+ experiment on a **local workstation or laptop** — no SLURM or supercomputer required.

> If you have not yet built the project, complete the [Installation](../installation.md) guide first.

---

## What This Tutorial Covers

- Running a single-array MVM experiment with a real device simulation.
- Understanding the output and MCA statistics.
- Running the same experiment inside Docker.
- Common next steps.

---

## The Experiment

We will run a distributed MVM on the `bcsstk02` matrix (included in the repo at `inputs/matrices/bcsstk02.mtx`) using a single simulated **EpiRAM** crossbar array with a 66×66 cell grid. This is the simplest possible configuration:

```
1 MCA (mca_rows=1, mca_cols=1)  →  2 MPI processes (1 worker + 1 root)
```

The configuration file for this experiment is already provided at:

```
config_files/quickstart/EpiRAM/exp1.yaml
```

---

## Step 1 — Activate Your Environment

```bash
cd meliso
conda activate mpienv38
export PYTHONPATH="${PYTHONPATH}:$(pwd)/build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:$(pwd)/build/"
```

---

## Step 2 — Inspect the Configuration File

Open `config_files/quickstart/EpiRAM/exp1.yaml`:

```yaml
exp_params:
  turnOnHardware: 1         # Enable hardware simulation
  turnOnScaling: 0          # No additional internal scaling
  matrix_name: "bcsstk02"  # Used for naming temp files
  matrix_file: "inputs/matrices/bcsstk02.mtx"
  distributed:
    decomposition_dir: "/tmp/"
    mca_rows: 1             # 1 row of MCA chips
    mca_cols: 1             # 1 column of MCA chips

device_config:
  root: "config_files/"
  cell_rows: 66             # 66 rows of cells per chip
  cell_cols: 66             # 66 cols of cells per chip
  assignment:
    device_EpiRAM.yaml: [[-1]]  # All chips use EpiRAM
```

Key things to note:
- `mca_rows: 1` and `mca_cols: 1` → **1 MCA chip** → needs **2 MPI processes**.
- `cell_rows: 66` and `cell_cols: 66` → the chip holds a 66×66 weight matrix.
- `device_EpiRAM.yaml: [[-1]]` → `-1` means all chips use this device config.

For a full explanation of every field, see the [Configuration Reference](../configuration.md).

---

## Step 3 — Run the Experiment

```bash
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

### What each variable does

| Variable | Value in this run | Description |
|----------|-------------------|-------------|
| `EXP_CONFIG_FILE` | `config_files/quickstart/EpiRAM/exp1.yaml` | Points to the experiment config |
| `XVEC_PATH` | `inputs/vectors/input_x.txt` | Input vector file |
| `-n 2` | 2 | `mca_rows × mca_cols + 1 = 1×1 + 1 = 2` |

---

## Step 4 — Understanding the Output

A successful run produces output similar to:

```
[INFO] Setting hardware ON
INFO: Device type set to 1 on rank 0
Experiment Configuration
{'turnOnHardware': 1, 'turnOnScaling': 0, ...}

[INFO]ROOT: begin virtualParallelMatVec at MCA 0,0
Computing MVM at Device Rank 0...
INFO: Rank = 0 : A_iter : 21, X_iter : 21: A_res: 0.00031, X_res: 0.00041
INFO: Elapsed Error Correction Time at Device Rank 0: 4.32

[INFO] MELISO+ Result (without normalization reversal):
 [0.412 0.387 0.501 ...]

[INFO] MELISO+ Result (with normalization reversal):
 [14.21 13.87 17.03 ...]

[INFO] Comparing memristive MVM result to CPU MVM result ...
[INFO] Relative L2 error:  0.0023
[INFO] Relative Loo error: 0.0041

MCAStats for Rank 0
    totalSubArrayArea = 3.21e-09
    subArrayIH->writeLatency = 0.00142
    subArrayIH->readLatency  = 4.51e-09
    ...
```

### Output explained

| Output | Meaning |
|--------|---------|
| `Device type set to 1 on rank 0` | Using `RealDevice` (RRAM) simulation |
| `A_iter: 21, X_iter: 21` | Write-and-verify ran 21 iterations for both the matrix and the vector |
| `A_res: 0.00031` | Final residual (difference between target and encoded weights). Smaller is better |
| `MELISO+ Result (without normalization reversal)` | MVM output in the normalised `[0,1]` domain |
| `MELISO+ Result (with normalization reversal)` | MVM output rescaled back to the original domain |
| `Relative L2 error` | How close the memristive result is to an ideal CPU computation. Values under `0.05` are generally good |
| `writeLatency` | Time taken to program the crossbar (seconds) |
| `readLatency` | Time taken to perform the MVM (seconds) |

---

## Step 5 — Saving Results to a Report File

Set `REPORT_PATH` to write the experiment summary to a file:

```bash
mkdir -p reports/quickstart/EpiRAM

EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
REPORT_PATH=reports/quickstart/EpiRAM/exp1_rep1.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

---

## Step 6 — Trying Other Devices and Configurations

### Swap the device material

Edit the `assignment` field in your YAML to use a different device file:

```yaml
  assignment:
    device_Ag-aSi.yaml: [[-1]]   # or device_TaOx-HfOx.yaml, device_AlOx-HfO2.yaml
```

See the [Configuration Reference](../configuration.md#device-materials) for a comparison of all four supported materials.

### Change the number of arrays (scale up)

To use a 3×3 grid of arrays with a larger cell size:

```yaml
exp_params:
  distributed:
    mca_rows: 3
    mca_cols: 3

device_config:
  cell_rows: 32
  cell_cols: 32
```

Then launch with **10 MPI processes** (`3×3 + 1 = 10`):

```bash
EXP_CONFIG_FILE=config_files/quickstart/exp2.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec -n 10 python3 DistributedMatVec.py
```

### Disable error correction

```bash
EC=0 EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

### Use an ideal device (no noise)

```bash
DT=0 EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

With `DT=0` (IdealDevice), the relative L2 error will be near zero — useful as a sanity check.

---

## Running Inside Docker

If you are using Docker (see [Installation — Option B](../installation.md#option-b-docker-installation)), the steps are identical except you need two additional MPI flags:

```bash
docker exec -it meliso bash

# Inside the container:
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
  mpiexec --allow-run-as-root --oversubscribe \
  -n 2 python3 DistributedMatVec.py
```

| Flag | Reason |
|------|--------|
| `--allow-run-as-root` | Docker containers typically run as root; OpenMPI refuses to run as root by default |
| `--oversubscribe` | Allows more MPI ranks than CPU cores, needed when Docker limits CPU resources |

---

## Environment Variable Summary for This Tutorial

| Variable | Used in this tutorial | Description |
|----------|-----------------------|-------------|
| `EXP_CONFIG_FILE` | Yes — required | Path to experiment YAML |
| `XVEC_PATH` | Yes | Path to input vector |
| `REPORT_PATH` | Optional | Path to write report output |
| `DT` | Optional (`0` for ideal) | Device type override |
| `EC` | Optional (`0` to disable) | Error correction flag |
| `ITER_LIMIT` | Optional | Write-and-verify iteration cap (default 21) |
| `OVERRIDE` | Optional | Force full `ITER_LIMIT` iterations |

---

## Next Steps

- **Run on a SLURM cluster**: See the [Distributed MVM Tutorial](run_distributed_MVM.md) for full SLURM shell script templates.
- **Run MLP inference**: See [`mlpInference.py`](../api/scripts/mlpInference.md) for running a two-layer neural network on memristive hardware.
- **Understand the configuration**: See the [Configuration Reference](../configuration.md) for every YAML field.
- **Understand the API**: See the [API Reference](../api/index.md) to work with MELISO+ programmatically.
