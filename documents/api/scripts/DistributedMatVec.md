# Driver Script: `DistributedMatVec.py`

**File:** `DistributedMatVec.py`
**Authors:** Huynh Quang Nguyen Vo (Oklahoma State University)

---

## Overview

`DistributedMatVec.py` is the main entry-point script for running a distributed matrix-vector multiplication (MVM) experiment with MELISO+. It demonstrates the complete workflow:

1. Load an input vector from disk.
2. Run the distributed MVM **without** min-max scaling reversal (result in `[0, 1]`).
3. Compare with a CPU benchmark.
4. Run the distributed MVM **with** min-max scaling reversal (result in original domain).
5. Compare with a CPU benchmark.

The script must be launched via `mpiexec` with at least 2 processes.

---

## Usage

```bash
# Minimal (1 hardware array, 2 MPI ranks):
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
  mpiexec -n 2 python3 DistributedMatVec.py

# With all options:
DT=1 OVERRIDE=1 ITER_LIMIT=21 \
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
REPORT_PATH=reports/quickstart/EpiRAM/exp1.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

---

## Script Flow

```python
# 1. Load input vector
xpath = os.environ.get("XVEC_PATH", "./inputs/vectors/input_x.txt")
xvec  = np.loadtxt(fname=xpath, delimiter=',')

# 2. Run 1 — no scaling reversal
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=False)
mv.finalize()
mv.acquireMCAStats()
y_minmax = mv.acquireResults()

mv.parallelizedBenchmarkMatVec(0, 0, correction=False)
mv.finalize()
mv.acquireMCAStats()

# 3. Run 2 — with scaling reversal
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=True)
mv.finalize()
mv.acquireMCAStats()
y_reversed_minmax = mv.acquireResults()

mv.parallelizedBenchmarkMatVec(0, 0, correction=True)
mv.finalize()
```

---

## Environment Variables

All environment variables are passed from the shell. Commonly set variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `EXP_CONFIG_FILE` | Experiment YAML path | **required** |
| `XVEC_PATH` | Input vector file | `./inputs/vectors/input_x.txt` |
| `DT` | Device type (0–6) | `1` |
| `ITER_LIMIT` | Write-and-verify iteration cap | `21` |
| `OVERRIDE` | Force full iteration count | `0` |
| `REPORT_PATH` | Output report file | `default_report.txt` |
| `TMPDIR` | Temp directory for intermediate files | `/tmp/` |

---

## Output

- **Console:** MVM result vectors, benchmark comparison, MCA statistics.
- **`<TMPDIR>/y_mem_result.txt`:** Saved memristive MVM result (used by benchmark).
- **`<REPORT_PATH>`:** Appended with configuration and MCA statistics.

---

## Dependencies

- `meliso` — compiled Cython extension
- `solver.matvec.MatVecSolver` — high-level MVM interface
- `numpy`
- `mpi4py`

---

## SLURM Example

```bash
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --mem=32GB

module load anaconda3/2022.10
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

export PYTHONPATH="${PYTHONPATH}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH}:./build/"

DT=1 OVERRIDE=0 ITER_LIMIT=21 \
EXP_CONFIG_FILE=config_files/quickstart/EpiRAM/exp1.yaml \
XVEC_PATH=inputs/vectors/input_x.txt \
REPORT_PATH=reports/run1.txt \
  mpiexec -n 2 python3 DistributedMatVec.py
```

---

## See Also

- [`MatVecSolver`](../solver/matvec/MatVecSolver.md) — The interface used by this script.
- [`mlpInference.py`](mlpInference.md) — MLP inference driver using the same framework.
- [Tutorials](../../tutorials/run_distributed_MVM.md) — Step-by-step guide.
