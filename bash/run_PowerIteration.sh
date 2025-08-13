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
JOB_TMPDIR=${JOB_TMPDIR:-/tmp/${SLURM_JOB_ID}}
mkdir -p "${JOB_TMPDIR}"
export MELISO_TMP_DIR="${JOB_TMPDIR}"
export TMPVEC_DIR="${JOB_TMPDIR}"

# ============================
# Device / simulation options used by MELISO+
#  DT         : Device Type seen by MELISO+ (0=Ideal, 1=Real)
#  OVERRIDE   : Write-and-verify behavior for synaptic weights
#               (0=adaptive stop, 1=force ITER_LIMIT)
#  ITER_LIMIT : Max write-and-verify iterations per crossbar array
#  PRECISION  : Programming granularity; residual tol ~ PRECISION^2
# ============================
export DT=${DT:-1}            # Default to Real Device
export OVERRIDE=${OVERRIDE:-0} # Default adaptive stop (will be set to 1 per-run when matrix is forced)
export ITER_LIMIT=${ITER_LIMIT:-21}
export PRECISION=${PRECISION:-1e-4}

# ============================
# Experiment grid
# ============================

# --- Number of replications (to account for device stochasticity) ---
REPS=${REPS:-1}

# --- Experiment IDs (to match YAML names like exp1.yaml) ---
EXPIDs=(${EXPIDs:-"1"})

# --- Materials mapping to a directory holding exp*.yaml config files ---
declare -A MATERIALS=(
  ["EpiRAM"]="${MATERIALS_EpiRAM:-config_files/power_iteration/EpiRAM}"
)

# --- Power Iteration options ---
MAX_ITERS_LIST=(${MAX_ITERS_LIST:-2000})
TOL=${TOL:-1e-6}
USE_CORRECTION=${USE_CORRECTION:-1}   # 1 -> apply error correction methods
SEED_LIST=(${SEED_LIST:-0})
SELFTEST=${SELFTEST:-0}               # 1 -> run --selftest (no matrix needed)

# --- Problems to process (space-separated list) ---
NPZ_FILES=(${NPZ_FILES:-inputs/problems/converted/relaxed_gen-ip002.npz})

# --- MPI tasks (defaults to SLURM request) ---
# IMPORTANT: NTASKS must equal (mca_rows * mca_cols + 1); MELISO root is the LAST rank.
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
    BASE_EXP_CFG="${CONFIG_DIR}/exp${expid}.yaml"

    for max_iter in "${MAX_ITERS_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        for npz in "${NPZ_FILES[@]}"; do

          # --- Build output locations ---
          if [[ "$SELFTEST" -eq 1 ]]; then
            tag="selftest"
          else
            tag="$(basename "${npz%.*}")_mi${max_iter}_seed${seed}"
          fi
          BASE_DIR="reports/PowerIteration/${material}/exp${expid}/${tag}"
          mkdir -p "${BASE_DIR}"

          REPORT_PATH="${BASE_DIR}/meliso_report.txt"
          CSV_AGG="${BASE_DIR}/power_iteration_results.csv"
          # For Python’s own temp vectors, keep them near outputs:
          TMPVEC_DIR="${BASE_DIR}/tmpvec_job${SLURM_JOB_ID}"

          # --- Per-job YAML copy with unique decomposition directory ---
          JOB_CONF="${BASE_DIR}/exp${expid}.yaml"
          cp -f "${BASE_EXP_CFG}" "${JOB_CONF}"

          # Create a job-scoped decomposition dir to avoid collisions
          DECOMP_DIR="${BASE_DIR}/decomposition"
          mkdir -p "${DECOMP_DIR}"
          sed -E -i.bak 's|(^[[:space:]]*decomposition_dir:[[:space:]]*).*|\1'"${JOB_TMPDIR}"'|' "${JOB_CONF}" || true

          # --- Clean old report (optional) ---
          [ -f "$REPORT_PATH" ] && rm -f "$REPORT_PATH"

          echo "============================================================"
          echo "Material:     ${material}"
          echo "Config base:  ${BASE_EXP_CFG}"
          echo "Config (job): ${JOB_CONF}"
          if [[ "$SELFTEST" -eq 1 ]]; then
            echo "Problem:      (selftest)"
          else
            echo "Problem:      ${npz}"
          fi
          echo "Max iters:    ${max_iter}"
          echo "Seed:         ${seed}"
          echo "Output dir:   ${BASE_DIR}"
          echo "Decomp dir:   ${DECOMP_DIR}"
          echo "Job tmp dir:  ${JOB_TMPDIR}"
          echo "NTASKS:       ${NTASKS}  (MELISO root = last rank)"
          echo "============================================================"

          export EXP_CONFIG_FILE="${JOB_CONF}"
          export REPORT_PATH

          # --- Compose arguments ---
          if [[ "$SELFTEST" -eq 1 ]]; then
            ARGS=(
              --selftest
              --reports-dir "$BASE_DIR"
              --tmpvec-dir "$TMPVEC_DIR"
            )
            [[ "$USE_CORRECTION" -eq 1 ]] && ARGS+=(--correction)
            unset A_FILE
          else
            # Lock MELISO to the matrix we pass to Python
            export A_FILE="$npz"
            export OVERRIDE=1
            echo "Forcing MELISO matrix: $A_FILE  (OVERRIDE=$OVERRIDE)"

            ARGS=(
              --matrix "$npz"
              --max-iter "$max_iter"
              --tol "$TOL"
              --reports-dir "$BASE_DIR"
              --seed "$seed"
              --save-temp-vectors
              --tmpvec-dir "$TMPVEC_DIR"
              # Optional safety nets:
              # --verify-matrix --strict-check --check-tol 5e-3
            )
            [[ "$USE_CORRECTION" -eq 1 ]] && ARGS+=(--correction)
          fi

          # --- Launch (the root process is last rank in MELISO+) ---
          mpiexec -n "$NTASKS" python3 PowerIteration_memristor.py "${ARGS[@]}"

        done
      done
    done
  done
done
