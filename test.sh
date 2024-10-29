#!/bin/bash
#SBATCH -p long
#SBATCH -t 12:00:00
#SBATCH --ntasks=1025
#SBATCH --cpus-per-task=1
#SBATCH --mail-user=lucius.vo@okstate.edu
#SBATCH --mail-type=end

echo "Running a test job on 2 nodes with 4 tasks each"
