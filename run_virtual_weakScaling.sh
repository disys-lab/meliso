#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 12:00:00
#SBATCH -n 260
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Enable debugging
set -x

# Removed 'set -e' to prevent the script from exiting on errors

# Setup
module load anaconda3/2022.10
module load gcc/7.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/

# Number of replications
REPS=10

# Number of iterations
ITER_LIMIT=(1 50 100)

# List of materials and corresponding config paths for each experiment
declare -A MATERIALS=(
    ["Ag-aSi"]="config_files/virtualization/weakScaling/Ag-aSi/"
    ["AlOx-HfO2"]="config_files/virtualization/weakScaling/AlOx-HfO2/"
    ["EpiRAM"]="config_files/virtualization/weakScaling/EpiRAM/"
    ["TaOx-HfOx"]="config_files/virtualization/weakScaling/TaOx-HfOx/"
)

# Experiment IDs
EXPIDs=("1.1" "2.1" "3.1")

# Map EXPIDs to NUM_PROCESSES
declare -A EXPID_TO_PROCESSES=(
    ["1.1"]=257
    ["2.1"]=257
    ["3.1"]=257
)

# Common input vector path
XVEC_PATH="inputs/vectors/input_x.txt"

# Loop over each material
for material in "${!MATERIALS[@]}"; do
    CONFIG_PATH="${MATERIALS[$material]}"

    # Loop over experiment IDs
    for expid in "${EXPIDs[@]}"; do
        EXP_CONFIG_FILE="${CONFIG_PATH}/exp${expid}.yaml"
        NUM_PROCESSES="${EXPID_TO_PROCESSES[$expid]}"

        # Check if the config file exists
        if [ ! -f "$EXP_CONFIG_FILE" ]; then
            echo "Warning: Config file $EXP_CONFIG_FILE does not exist. Skipping."
            continue
        fi

        # Loop over each ITER_LIMIT
        for iter_limit in "${ITER_LIMIT[@]}"; do

            # Run the experiment REPS times for each ITER_LIMIT
            for ((i=1; i<=REPS; i++)); do
                echo "Running ${material}, exp${expid} with ITER_LIMIT=${iter_limit}, repetition $i"
                REPORT_PATH="reports/virtualization/weakScaling/${material}/exp${expid}_iter_${iter_limit}_rep_${i}.txt"

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
                mpiexec -n "$NUM_PROCESSES" python3 DistributedMatVec.py || echo "Error running mpiexec for $material, exp${expid}, iteration $iter_limit, repetition $i"
            done
        done
    done
done
