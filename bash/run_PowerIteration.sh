#!/bin/bash
#SBATCH -p express
#SBATCH -t 1:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END

# Exit immediately on error
set -e

# ============================
# Modules and Conda Distribution
# ============================
module load anaconda3/2022.10
module load gcc/7.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# ============================
# Paths and Environments for MELISO+ & MLP+NeuroSim Backend
# ============================
export PYTHONPATH="${PYTHONPATH:-}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"

# ============================
# Device / simulation options used by MELISO+
#  DT         : Device Type seen by MELISO+ (0=Ideal, 1=Real)
#  OVERRIDE   : Write-and-verify behavior for synaptic weights (0=adaptive stop, 1=force ITER_LIMIT)
#  ITER_LIMIT : Maximum write-and-verify iterations per crossbar array
#  PRECISION  : Programming granularity; residual tolerance ~ PRECISION^2
# ============================
export DT=${DT:-1}  # Default to Real Device
export OVERRIDE=${OVERRIDE:-0} # Default to adaptive stop
export ITER_LIMIT=${ITER_LIMIT:-21} # Default to 21 iterations
export PRECISION=${PRECISION:-1e-4} # Default to 1e-4 precision

# ============================
# Experiment grid
# ============================

# --- Number of replications for the experiment (to account for the memristor's stochastic behaviours) ---
REPS=${REPS:-1} # Default to 1 replication

# --- Experiment IDs (to match YAML names like exp1.yaml) ---
EXPIDs=(${EXPIDs:-"1"})

# --- Materials mapping to a directory holding exp*.yaml config files ---
declare -A MATERIALS=(
  ["EpiRAM"]="${MATERIALS_EpiRAM:-config_files/power_iteration/EpiRAM}"
  )

# --- Power Iteration options ---
MAX_ITERS_LIST=(${MAX_ITERS_LIST:-2000})
TOL=${TOL:-1e-6}
USE_CORRECTION=${USE_CORRECTION:-1}   # Default to 1 -> apply error correction methods
SEED_LIST=(${SEED_LIST:-0}) # Default to seed 0
SELFTEST=${SELFTEST:-0} # Default to 0 -> do not run self-tests

# --- Problems to process (space-separated list) ---
# Adjust the path as needed
NPZ_FILES=(${NPZ_FILES:-inputs/problems/converted/relaxed_gen-ip002.npz})

# --- MPI tasks (defaults to SLURM request) ---
NTASKS=${NTASKS:-${SLURM_NTASKS:-2}}

# --- If self-test is enabled, run once with a dummy placeholder (no matrix needed) ---
if [[ "$SELFTEST" -eq 1 ]]; then
  NPZ_FILES=("__selftest__")
fi

# ============================
# Main Execution
# ============================
for material in "${!MATERIALS[@]}"; do
  CONFIG_DIR="${MATERIALS[$material]}"

  for expid in "${EXPIDs[@]}"; do
    EXP_CONFIG_FILE="${CONFIG_DIR}/exp${expid}.yaml"

    for max_iter in "${MAX_ITERS_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        for npz in "${NPZ_FILES[@]}"; do

          # --- Build output locations ---
          tag="$(basename "${npz%.*}")_mi${max_iter}_seed${seed}"
          BASE_DIR="reports/PowerIteration/${material}/exp${expid}/${tag}"
          mkdir -p "${BASE_DIR}"

          REPORT_PATH="${BASE_DIR}/meliso_report.txt"
          CSV_AGG="${BASE_DIR}/power_iteration_results.csv"
          TMPVEC_DIR="${BASE_DIR}/tmpvec_job${SLURM_JOB_ID}"

          # --- Clean old report (optional) ---
          [ -f "$REPORT_PATH" ] && rm -f "$REPORT_PATH"

          echo "============================================================"
          echo "Material:   ${material}"
          echo "Config:     ${EXP_CONFIG_FILE}"
          echo "Problem:    ${npz}"
          echo "Max iters:  ${max_iter}"
          echo "Seed:       ${seed}"
          echo "Output dir: ${BASE_DIR}"
          echo "============================================================"

          # --- Export MELISO+-related environment for the backend ---
          export EXP_CONFIG_FILE REPORT_PATH

          # --- Compose arguments ---
          if [[ "$SELFTEST" -eq 1 ]]; then
            ARGS=(
              --selftest
              --reports-dir "$BASE_DIR"
            )
            if [[ "$USE_CORRECTION" -eq 1 ]]; then
              ARGS+=(--correction)
            fi
          else
            ARGS=(
              --matrix "$npz"
              --max-iter "$max_iter"
              --tol "$TOL"
              --reports-dir "$BASE_DIR"
              --seed "$seed"
              --save-temp-vectors
              --tmpvec-dir "$TMPVEC_DIR"
            )
            if [[ "$USE_CORRECTION" -eq 1 ]]; then
              ARGS+=(--correction)
            fi
          fi

          # --- Launch (the root process is last rank in MELISO+) ---
          mpiexec -n "$NTASKS" python3 PowerIteration_memristor.py "${ARGS[@]}"
        done
      done
    done
  done
done
