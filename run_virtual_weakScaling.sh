#!/bin/bash

# Setup
export PYTHONPATH=$PYTHONPATH:./build
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:./build/
python3 MelisoDriver.py

# Number of replications
REPS=5

# List of materials and corresponding config paths for each experiment
declare -A MATERIALS
MATERIALS=( 
    ["Ag-aSi"]="config_files/virtualization/weakScaling/Ag-aSi/"
    ["AlOx-HfO2"]="config_files/virtualization/weakScaling/AlOx-HfO2/"
    ["EpiRAM"]="config_files/virtualization/weakScaling/EpiRAM/"
    ["TaOx-HfOx"]="config_files/virtualization/weakScaling/TaOx-HfOx/"
)

# List of experiment file names and corresponding number of processors
EXPERIMENTS=("exp1.yaml" "exp2.yaml" "exp3.yaml" "exp4.yaml")
PROCESSORS=(1025 257 65 17)  # Number of processors for each experiment

# Constant ITER_LIMIT value
ITER_LIMIT=100

# Common input vector path
XVEC_PATH="inputs/vectors/input_x.txt"

# Loop over each material
for material in "${!MATERIALS[@]}"; do

  # Loop over each experiment configuration
  for idx in "${!EXPERIMENTS[@]}"; do
    EXP_CONFIG_FILE="${MATERIALS[$material]}${EXPERIMENTS[$idx]}"
    NUM_PROCESSORS="${PROCESSORS[$idx]}"  # Get the corresponding number of processors

    # Run the experiment REPS times with the constant ITER_LIMIT
    for ((i=1; i<=REPS; i++)); do
      echo "Running ${material} with ${EXPERIMENTS[$idx]}, ITER_LIMIT=${ITER_LIMIT}, repetition $i using ${NUM_PROCESSORS} processors"
      REPORT_PATH="exp_reports/virtualization/weakScaling/${material}/${EXPERIMENTS[$idx]%.yaml}_iter_${ITER_LIMIT}_rep_${i}.txt"
      
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
