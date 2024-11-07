#!/bin/bash
#SBATCH -p cascadelake
#SBATCH -t 12:00:00
#SBATCH -n 10
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=END
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Setup
module load anaconda3/2022.10
module load gcc/7.5.0
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate mpienv38

# Set environment variables
export DT=1
export OVERRIDE=1
export ITER_LIMIT=100
export XVEC_PATH=inputs/vectors/input_x.txt
export EXP_CONFIG_FILE=config_files/virtualization/test/Ag-aSi/exp1.1.yaml

# Run the experiment 10 times
for i in {1..10}
do
    echo "Running experiment iteration $i..."
    export REPORT_PATH="reports/virtualization/test/Ag-aSi/exp1.1_run_$i.txt"
    mpiexec -n 10 python3 DistributedMatVec.py
    echo "Completed iteration $i."
done

echo "All 10 experiment iterations completed."
