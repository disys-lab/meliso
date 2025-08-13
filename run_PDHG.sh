#!/bin/bash
#SBATCH -p batch
#SBATCH -t 24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END

# Exit immediately on error
set -e

# --- Modules & Conda ---
module load anaconda3/2022.10
module load gcc/7.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# --- Paths / Env for MELISO & C++ backend ---
export PYTHONPATH="${PYTHONPATH:-}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"

# Device / simulation options used by MELISO+
export DT=${DT:-1}             # device type seen by MELISO+ (1 = RealDevice in your stack)
export OVERRIDE=${OVERRIDE:-0} # write and verify seen by MELISO+ (0 = Stop the process when the residuals are too small)
export ITER_LIMIT=${ITER_LIMIT:-100}   # max programming iterations per tile
export PRECISION=${PRECISION:-1e-8}    # programming step; residual tol ~ PRECISION^2

# === Experiment grid ===
# Number of replications
REPS=${REPS:-1}

# Experiment IDs (match YAML names like exp1.yaml)
EXPIDs=(${EXPIDs:-"1"})

# Materials -> config dir
declare -A MATERIALS=(
    ["EpiRAM"]="${MATERIALS_EpiRAM:-config_files/pdhg/EpiRAM}"
)

# PDHG options
MAX_ITERS_LIST=(${MAX_ITERS_LIST:-200000})
TOL=${TOL:-1e-6}
THETA=${THETA:-1.0}
NORM_ITERS=${NORM_ITERS:-200}
PATIENCE_LIST=(${PATIENCE_LIST:-5000})
MONITOR=${MONITOR:-gap}          # gap | primal_dual | kkt
MIN_DELTA=${MIN_DELTA:-0.0}
PLOT_EVERY=${PLOT_EVERY:-500}
USE_CORRECTION=${USE_CORRECTION:-1}  # 1 -> pass --correction

# Problems (NPZ files). Space-separated list; customize as needed.
NPZ_FILES=(${NPZ_FILES:-inputs/problems/converted/relaxed_gen-ip002.npz})

# MPI tasks (defaults to Slurm request)
NTASKS=${NTASKS:-${SLURM_NTASKS:-2}}

# --- Run ---
for material in "${!MATERIALS[@]}"; do
    CONFIG_PATH="${MATERIALS[$material]}"

    for expid in "${EXPIDs[@]}"; do
        EXP_CONFIG_FILE="${CONFIG_PATH}/exp${expid}.yaml"

        for max_iter in "${MAX_ITERS_LIST[@]}"; do
          for patience in "${PATIENCE_LIST[@]}"; do
            for npz in "${NPZ_FILES[@]}"; do

                # Build output locations
                tag="$(basename "${npz%.*}")_mi${max_iter}_pat${patience}_${MONITOR}"
                BASE_DIR="reports/PDHG/${material}/exp${expid}/${tag}"
                mkdir -p "${BASE_DIR}"

                REPORT_PATH="${BASE_DIR}/meliso_report.txt"
                CSV_LOG="${BASE_DIR}/diag_logs.csv"
                SOL_PATH="${BASE_DIR}/x.npy"
                PLOT_PREFIX="${BASE_DIR}/diag"

                # Clean old report (optional)
                [ -f "$REPORT_PATH" ] && rm -f "$REPORT_PATH"

                echo "============================================================"
                echo "Material:   ${material}"
                echo "Config:     ${EXP_CONFIG_FILE}"
                echo "Problem:    ${npz}"
                echo "Max iters:  ${max_iter}"
                echo "Patience:   ${patience} (monitor=${MONITOR}, min_delta=${MIN_DELTA})"
                echo "Plots/logs: ${BASE_DIR}"
                echo "============================================================"

                # Export MELISO-related env for the MCA backend
                export EXP_CONFIG_FILE REPORT_PATH

                # Compose arguments
                ARGS=(
                  --npz "$npz"
                  --max_iter "$max_iter"
                  --tol "$TOL"
                  --theta "$THETA"
                  --norm_iters "$NORM_ITERS"
                  --plot_prefix "$PLOT_PREFIX"
                  --plot_every "$PLOT_EVERY"
                  --save_logs "$CSV_LOG"
                  --save_solution "$SOL_PATH"
                  --patience "$patience"
                  --monitor "$MONITOR"
                  --min_delta "$MIN_DELTA"
                )
                if [[ "$USE_CORRECTION" -eq 1 ]]; then
                    ARGS+=(--correction)
                fi

                # Launch (root is last rank in your MELISO code path)
                mpiexec -n "$NTASKS" python3 pdhg_memristor_tweak.py "${ARGS[@]}"
            done
          done
        done
    done
done
