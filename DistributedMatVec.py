"""
@author: Paritosh Ramanan, Huynh Quang Nguyen Vo

This script demonstrates the usage of the MPI-based distributed matrix-vector multiplication
(MVM) solver on MELISO+. It initializes the solver, runs the MVM operation, finalizes the solver, 
acquires MCA statistics, runs a benchmark, and then finalizes again.

Usage:
    Run this script in an MPI environment.
"""

import meliso
import os
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

# --------------------------------------------------------------------------------------------------
# Load input vectors
# --------------------------------------------------------------------------------------------------
xpath = os.environ.get("XVEC_PATH", "./inputs/vectors/input_x.txt")
xvec = np.loadtxt(fname=xpath, delimiter=',')

# --------------------------------------------------------------------------------------------------
# Run the distributed matrix-vector multiplication (MVM) with no min-max scaling reversion and 
# compare with the benchmark results
# --------------------------------------------------------------------------------------------------
CORRECTION = False # 
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=False) 
mv.finalize() # Memristive MVM should be in the [0,1] range
mv.acquireMCAStats() 

mv.parallelizedBenchmarkMatVec(0,0,correction = CORRECTION) # The benchmarking results should also be in the [0,1] range
mv.finalize()

# --------------------------------------------------------------------------------------------------
# Run the distributed matrix-vector multiplication (MVM) with min-max scaling reversion and 
# compare with the benchmark results
# --------------------------------------------------------------------------------------------------
CORRECTION = True # With min-max scaling reversion 
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=True) 
mv.finalize() # Memristive MVM should be in the original range
mv.acquireMCAStats() 

mv.parallelizedBenchmarkMatVec(0,0,correction = CORRECTION) # The benchmarking results should also be in the original range
mv.finalize()