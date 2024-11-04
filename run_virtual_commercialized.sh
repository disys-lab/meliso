#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 12:00:00
#SBATCH --ntasks=1030
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END

# Exit immediately if a command exits with a non-zero status
set -e

# Setup
module load anaconda3/2022.10
module load gcc/7.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# Set up environment variables
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/

# Number of replications
REPS=5

# Number of iterations
ITER_LIMIT=10

# List of materials and corresponding config paths for each experiment
declare -A MATERIALS
MATERIALS=( 
    ["Ag-aSi"]="config_files/virtualization/commercialized/Ag-aSi/"
    ["AlOx-HfO2"]="config_files/virtualization/commercialized/AlOx-HfO2/"
    ["EpiRAM"]="config_files/virtualization/commercialized/EpiRAM/"
    ["TaOx-HfOx"]="config_files/virtualization/commercialized/TaOx-HfOx/"
)

# List of experiment file names and corresponding number of processors
EXPERIMENTS=("exp1.yaml" "exp2.yaml" "exp3.yaml" "exp4.yaml" "exp5.yaml" "exp6.yaml")
PROCESSORS=10

# Common input vector path
XVEC_PATH="inputs/vectors/input_x.txt"

# Loop over each material
for material in "${!MATERIALS[@]}"; do

  # Loop over each experiment configuration
  for idx in "${!EXPERIMENTS[@]}"; do
    EXP_CONFIG_FILE="${MATERIALS[$material]}${EXPERIMENTS[$idx]}"

    # Run the experiment REPS times with the constant ITER_LIMIT
    for ((i=1; i<=REPS; i++)); do
      echo "Running ${material} with ${EXPERIMENTS[$idx]}, ITER_LIMIT=${ITER_LIMIT}, repetition $i using ${NUM_PROCESSORS} processors"
      REPORT_PATH="reports/virtualization/weakScaling/${material}/${EXPERIMENTS[$idx]%.yaml}_iter_${ITER_LIMIT}_rep_${i}.txt"
      
      # Remove old report file if it exists
      if [ -f "$REPORT_PATH" ]; then
        echo "Removing old report file: $REPORT_PATH"
        rm "$REPORT_PATH"
      fi

      # Run the experiment
      DT=1 OVERRIDE=1 ITER_LIMIT=$ITER_LIMIT XVEC_PATH=$XVEC_PATH \
      EXP_CONFIG_FILE=$EXP_CONFIG_FILE REPORT_PATH=$REPORT_PATH \
      mpiexec -n $NUM_PROCESSORS python3 DistributedMatVec.py
    done
  done
done
