#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 12:00:00
#SBATCH -n 10
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
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/

# Set environment variables
export DT=1
export OVERRIDE=1
export ITER_LIMIT=100

export XVEC_PATH=inputs/vectors/input_x.txt

# Define arrays of materials and experiment IDs
MATERIALS=("Ag-aSi" "AlOx-HfO2" "EpiRAM" "TaOx-HfOx")
EXPERIMENTS=("exp1.1" "exp1.2")

# Loop over materials and experiments
for MATERIAL in "${MATERIALS[@]}"
do
    for EXP_ID in "${EXPERIMENTS[@]}"
    do
        # Set the experiment configuration file
        export EXP_CONFIG_FILE="config_files/test/${MATERIAL}/${EXP_ID}.yaml"
        
        # Create report directory if it doesn't exist
        REPORT_DIR="reports/test/${MATERIAL}"
        mkdir -p "${REPORT_DIR}"
        
        # Run the experiment 10 times
        for i in {1..10}
        do
            echo "Running ${MATERIAL} ${EXP_ID} iteration $i..."
            export REPORT_PATH="${REPORT_DIR}/${EXP_ID}_run_${i}.txt"
            mpiexec -n 10 python3 DistributedMatVec.py
            echo "Completed iteration $i of ${MATERIAL} ${EXP_ID}."
        done
    done
done

echo "All experiments completed."
