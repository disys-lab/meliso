#===================================================================================================
# DRIVER SCRIPT FOR DISTRIBUTED MATRIX-VECTOR MULTIPLICATION (MVM)
#===================================================================================================
"""
@author: Huynh Quang Nguyen Vo
@affiliation: Oklahoma State University
This script demonstrates how to perform distributed matrix-vector multiplication (MVM) using the 
MELISO+ framework. It loads input vectors, runs the MVM with and without min-max scaling correction,
and compares the results with benchmark outputs.
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
y_minmax = mv.acquireResults()
print("Obtained MVM result without min-max scaling correction: \n", y_minmax)

mv.parallelizedBenchmarkMatVec(0,0,correction = CORRECTION) # The benchmarking results should also be in the [0,1] range
mv.finalize()
mv.acquireMCAStats()

# --------------------------------------------------------------------------------------------------
# Run the distributed matrix-vector multiplication (MVM) with min-max scaling reversion and 
# compare with the benchmark results
# --------------------------------------------------------------------------------------------------
CORRECTION = True # With min-max scaling reversion 
mv = MatVecSolver(xvec=xvec)
mv.matVec(correction=True) 
mv.finalize() # Memristive MVM should be in the original range
mv.acquireMCAStats() 
y_reversed_minmax = mv.acquireResults()
print("Obtained MVM result with min-max scaling correction: \n", y_reversed_minmax)

mv.parallelizedBenchmarkMatVec(0,0,correction = CORRECTION) # The benchmarking results should also be in the original range
mv.finalize()