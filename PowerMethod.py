from mpi4py import MPI
import numpy as np
import os
from solver.matvec.MatVecSolver import MatVecSolver
from typing import List, Tuple

class PowerIteration:
    """
    @author: vohuynhquangnguyen
    """

    RESULT_FILENAME = "y_mem_result.csv"
    X_ITERATES_FILENAME = "x_iterates.csv"
    LOG_FILENAME = "x_log.txt"

    def __init__(self, A: np.ndarray, x_init: np.ndarray, lam_init: float,
                 num_iterations: int, tol: float) -> None:
        self.A = A
        self.x = x_init
        self.lam = lam_init
        self.num_iterations = num_iterations
        self.tol = tol
    
    # --- Matrix-Vector Multiplication (MVM) ---
    def _compute_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Helper method to compute matrix-vector product using external solver."""
        self.mv_solver.solverObject.initializeMat(matrix)
        self.mv_solver.solverObject.initializeX(vector)
        self.mv_solver.matVec(correction=True)
        return np.loadtxt(self.RESULT_FILENAME, delimiter=",")

    def solve(self):
        for _ in range(self.num_iterations):
            A_x = self._compute_matvec(self.A, self.x_init)






