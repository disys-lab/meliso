#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 12:00:00
#SBATCH -n 260
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
# Exit immediately if a command exits with a non-zero status
set -e

# Setup
module load anaconda3/2022.10
module load gcc/7.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# Set up environment variables
export PYTHONPATH="${PYTHONPATH:-}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"

# Number of replications
REPS=10

# Experiment IDs
EXPIDs=("1" "2")

# List of materials and corresponding config paths
declare -A MATERIALS=(
    ["Ag-aSi"]="config_files/iterations/Ag-aSi"
    ["AlOx-HfO2"]="config_files/iterations/AlOx-HfO2"
    ["EpiRAM"]="config_files/iterations/EpiRAM"
    ["TaOx-HfOx"]="config_files/iterations/TaOx-HfOx"
)

# List of ITER_LIMIT values
ITER_LIMITS=(1 10 20 30 40 50 60 70 80 90 100)

# Common input vector path
XVEC_PATH="inputs/vectors/input_x.txt"

# Loop over each material and experiment ID, then run experiments
for material in "${!MATERIALS[@]}"; do
    CONFIG_PATH="${MATERIALS[$material]}"

    for expid in "${EXPIDs[@]}"; do
        EXP_CONFIG_FILE="${CONFIG_PATH}/exp${expid}.yaml"

        # Loop over each ITER_LIMIT
        for iter_limit in "${ITER_LIMITS[@]}"; do

            # Run the experiment REPS times for each ITER_LIMIT
            for ((i=1; i<=REPS; i++)); do
                echo "Running ${material}, exp${expid} with ITER_LIMIT=${iter_limit}, repetition $i"
                REPORT_PATH="reports/iterations/${material}/exp${expid}_iter_${iter_limit}_rep_${i}.txt"

                # Create the report directory if it doesn't exist
                mkdir -p "$(dirname "$REPORT_PATH")"

                # Remove old report file if it exists
                if [ -f "$REPORT_PATH" ]; then
                    echo "Removing old report file: $REPORT_PATH"
                    rm "$REPORT_PATH"
                fi

                # Run the experiment
                DT=1 OVERRIDE=1 ITER_LIMIT="$iter_limit" XVEC_PATH="$XVEC_PATH" \
                EXP_CONFIG_FILE="$EXP_CONFIG_FILE" REPORT_PATH="$REPORT_PATH" \
                mpiexec -n 2 python3 DistributedMatVec.py
            done
        done
    done
done
