#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 24:00:00
#SBATCH -n 10 
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
export PYTHONPATH="${PYTHONPATH:-}:./build"
export LD_LIBRARY_PATH="${LD_LIBRARY_PATH:-}:./build/"

# Number of replications
REPS=10

# Experiment IDs
EXPIDs=("1")

# List of materials and corresponding config paths
declare -A MATERIALS=(
    ["Ag-aSi"]="config_files/virtualization/commercialized/Ag-aSi"
    ["AlOx-HfO2"]="config_files/virtualization/commercialized/AlOx-HfO2"
    ["EpiRAM"]="config_files/virtualization/commercialized/EpiRAM"
    ["TaOx-HfOx"]="config_files/virtualization/commercialized/TaOx-HfOx"
)
