#!/bin/bash

# Number of replications
REPS=5

# List of materials and corresponding config paths
declare -A MATERIALS
MATERIALS=(
    ["Ag-aSi"]="config_files/iterations/Ag-aSi/exp2.yaml"
    ["AlOx-HfO2"]="config_files/iterations/AlOx-HfO2/exp2.yaml"
    ["EpiRAM"]="config_files/iterations/EpiRAM/exp2.yaml"
    ["TaOx-HfOx"]="config_files/iterations/TaOx-HfOx/exp2.yaml"
)

# List of ITER_LIMIT values
ITER_LIMITS=(1 3 5 10 20 50 100)

# Common input vector path
XVEC_PATH="inputs/vectors/input_x.txt"

# Loop over each material and run experiments
for material in "${!MATERIALS[@]}"; do
  EXP_CONFIG_FILE="${MATERIALS[$material]}"

  # Loop over each ITER_LIMIT
  for iter_limit in "${ITER_LIMITS[@]}"; do

    # Run the experiment REPS times for each ITER_LIMIT
    for ((i=1; i<=REPS; i++)); do
      echo "Running ${material} with ITER_LIMIT=${iter_limit}, repetition $i"
      REPORT_PATH="exp_reports/iterations/${material}/exp2_iter_${iter_limit}_rep_${i}.txt"
      
      # Remove old report file if it exists
      if [ -f "$REPORT_PATH" ]; then
        echo "Removing old report file: $REPORT_PATH"
        rm "$REPORT_PATH"
      fi

      # Run the experiment
      DT=1 OVERIDE=1 ITER_LIMIT=$iter_limit XVEC_PATH=$XVEC_PATH \
      EXP_CONFIG_FILE=$EXP_CONFIG_FILE REPORT_PATH=$REPORT_PATH \
      mpiexec -n 2 python DistributedMatVec.py
    done

  done
done
