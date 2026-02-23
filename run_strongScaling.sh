#!/bin/bash

#===================================================================================================
# SCRIPT DESCRIPTION
#===================================================================================================
# This script automates running a series of computational experiments using the
# 'MELISO_MLP.py' Python script. It is designed to be submitted to a Slurm workload
# manager.
#
# The script iterates through a predefined set of materials, experiment IDs, and
# iteration limits. For each unique combination, it runs the simulation multiple times
# (replications) and saves the output to a unique report file.
#
# Key operations:
#   1. Sets up the necessary software environment (Anaconda, GCC).
#   2. Defines experiment parameters in a dedicated configuration section.
#   3. Loops through all parameter combinations.
#   4. For each run, it creates a unique output directory and report file.
#   5. Executes the Python script in parallel using 'mpiexec'.
#===================================================================================================


#===================================================================================================
# SLURM DIRECTIVES -- Job scheduler settings
#===================================================================================================
#SBATCH --partition long                          # Partition (queue) to submit the job to
#SBATCH --time 500:00:00                          # Maximum runtime in HH:MM:SS format
#SBATCH --ntasks=65                               # Total MPI ranks
#SBATCH --mem=72GB                                # Total memory required for the job
#SBATCH --mail-user=lucius.vo@okstate.edu         # Email address for job notifications
#SBATCH --mail-type=END                           # Send an email when the job finishes
#SBATCH --job-name=strongScalingMatVec              # Job name for easier identification
#===================================================================================================


#===================================================================================================
# ENVIRONMENT SETUP
#===================================================================================================
echo "[INFO] Setting up the environment..."

# Load required modules
module load anaconda3/2022.10
module load gcc/7.5.0

# Activate the specific conda environment
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# Add the local 'build' directory to Python's search path and the dynamic linker's path
export PYTHONPATH="${PYTHONPATH:-}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"

echo "[INFO] Environment setup complete."
#===================================================================================================


#===================================================================================================
# SCRIPT BEHAVIOR & ERROR HANDLING
#===================================================================================================
# 'set -e' will cause the script to exit immediately if any command fails.
# 'set -u' will treat unset variables as an error, preventing unexpected behavior.
# 'set -o pipefail' ensures that a pipeline's exit code is the status of the last command to exit
# with a non-zero status, or zero if no command failed.
set -euo pipefail
#===================================================================================================


#===================================================================================================
# EXPERIMENT CONFIGURATION
#===================================================================================================
EXPERIMENT_NAME="strongScaling"           # A name for this set of experiments, used in report paths.
NUM_PROCESSES=65                        # Total number of MPI processes to use for each run.
DEVICE_TYPE=1                           # Device type to use for the experiments.
NUM_REPLICATIONS=100                    # Number of repetitions for each experiment configuration
EXPERIMENT_IDS=("1" "2" "3" "4" "5" "6")    # A list of experiment IDs to run.
ITERATION_LIMITS=(21)                       # A list of iteration limits for the write-and-verify.
ENABLE_OVERRIDE=0                           # Whether to enable the override feature for iteration limits
INPUT_VECTOR_PATH="inputs/vectors/input_x.txt" # The file path for the common input vector.

# Define the materials to be tested and the paths to their configuration directories.
declare -A MATERIAL_CONFIGS=(
    ["Ag-aSi"]="config_files/${EXPERIMENT_NAME}/Ag-aSi"
    ["AlOx-HfO2"]="config_files/${EXPERIMENT_NAME}/AlOx-HfO2"
    ["EpiRAM"]="config_files/${EXPERIMENT_NAME}/EpiRAM"
    ["TaOx-HfO2"]="config_files/${EXPERIMENT_NAME}/TaOx-HfO2"
    )


#===================================================================================================
# EXPERIMENT EXECUTION LOOP
#===================================================================================================
echo "[INFO] Starting experiment execution..."

# Loop over each material name defined in the MATERIAL_CONFIGS array.
for material in "${!MATERIAL_CONFIGS[@]}"; do
    CONFIG_PATH="${MATERIAL_CONFIGS[$material]}"

    # Loop over each experiment ID.
    for expid in "${EXPERIMENT_IDS[@]}"; do
        EXP_CONFIG_FILE="${CONFIG_PATH}/exp${expid}.yaml"

        # Loop over each specified iteration limit.
        for iter_limit in "${ITERATION_LIMITS[@]}"; do

            # Run the same experiment configuration multiple times for statistical robustness.
            for ((rep=1; rep<=NUM_REPLICATIONS; rep++)); do

                # --- PREPARE FOR THIS RUN ---
                echo "[INFO] RUNNING: Material='${material}', Exp_ID='${expid}', Iter_Limit='${iter_limit}', Repetition='${rep}'"


                # --- EXECUTE THE SIMULATION ---
                RUN_TAG="${material}_exp${expid}_it${iter_limit}_rep${rep}"
                REPORT_PATH="$TMPDIR/logs/${RUN_TAG}.log"
                echo "[INFO]   - Config File: ${EXP_CONFIG_FILE}"
                echo "[INFO]   - Report File: ${REPORT_PATH}"

                # Set environment variables for the Python script and run it with mpiexec.
                # These variables are only set for the duration of this single command.
                DT=1 OVERRIDE=0 ITER_LIMIT="$iter_limit" XVEC_PATH="$INPUT_VECTOR_PATH" \
                EXP_CONFIG_FILE="$EXP_CONFIG_FILE" REPORT_PATH="$REPORT_PATH" \
                mpiexec -n 2 python3 MELISO_MLP.py

                echo "[INFO] DONE: Repetition ${rep} finished."
                echo -e "\n"

            done # End of replications loop
        done # End of iteration limits loop
    done # End of experiment IDs loop
done # End of materials loop

echo "[INFO] All experiments completed successfully!"