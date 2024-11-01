#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 12:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=32
#SBATCH --cpus-per-task=1
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
export PYTHONPATH="$PYTHONPATH:./build"
export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:./build/"
export DT=1
export OVERRIDE=1
export ITER_LIMIT=1
export XVEC_PATH="inputs/vectors/input_x.txt"
export EXP_CONFIG_FILE="config_files/quickstart/exp1.yaml"
export REPORT_PATH="reports/quickstart/exp1.txt"

# Run the serial job
python3 MelisoDriver.py

# Run the parallel job using srun for better integration with SLURM
mpiexec -n 2 python3 DistributedMatVec.py

# Run the serial job to get matrices' properties
python3 preprocessing.py
