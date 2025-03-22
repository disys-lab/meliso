"""
@author: Paritosh Ramanan, Huynh Quang Nguyen Vo
DistributedMatVec Module

This script demonstrates the usage of the MPI-based distributed matrix-vector multiplication
(MVM) solver on MELISO. It initializes the solver, runs a 
matrix-vector multiplication, finalizes the solver, acquires MCA statistics, runs a 
benchmark, and then finalizes again.

Usage:
    Run this script in an MPI environment.
"""
import time
import numpy as np
from scipy.io import mmread
from solver.matvec.MatVecSolver import MatVecSolver

def main():
    # --- Global constants ---
    correction = True
    A = mmread("inputs/matrices/Iperturb.mtx")
    x = np.loadtxt("inputs/vectors/input_x.txt")

    start_time = time.time()
    # --- Distributed MVM ---
    solver = MatVecSolver()
    solver.initialize_data(A, x)
    solver.matvec_mul(correction=correction)
    solver.finalize()
    solver.acquire_mca_stats()
    solver.benchmark_matvec_mul(hardware_on=0, scaling_on=0, correction=correction)
    solver.finalize()

    end_time = time.time()
    print(f"Total execution time: {end_time - start_time:.2f} seconds.")

if __name__ == "__main__":
    main()
