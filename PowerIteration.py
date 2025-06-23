"""
@author: Huynh Quang Nguyen Vo
Power Iteration Module

This script demonstrates the usage of the MPI-based distributed power iteration method on MELISO.

Usage:
    Run this script in an MPI environment.
"""

import time
import numpy as np
from scipy.io import mmread
from solver.matvec.MatVecSolver import MatVecSolver
from typing import Tuple

class PowerIteration:
    """
    Power Iteration Method for computing the dominant eigenvalue.
    """
    RESULT_FILENAME = "y_mem_result.csv"

    def __init__(self, A: np.ndarray, num_iterations: int, tol: float, matrix_name:str = None):
        self.A = A
        self.num_iterations = num_iterations
        self.tol = tol
        self.matrix_name = matrix_name

        # --- Instantiate the MatVecSolver object. ---
        self.mv_solver = MatVecSolver()
    
    def _compute_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Helper method to compute matrix-vector product using the MELISO MatVecSolver.
        """
        self.mv_solver.initialize_data(matrix, vector)
        self.mv_solver.matvec_mul(correction=True)
        self.mv_solver.finalize()
        self.mv_solver.acquire_mca_stats()
        return np.loadtxt(self.RESULT_FILENAME, delimiter=",")
    
    def solve(self) -> float:
        """
        Perform the two-sided power iteration and return the approximate spectral norm.
        """
        # --- Initialize a normalized random vector ---
        v = np.random.rand(self.A.shape[1])
        v /= np.linalg.norm(v)

        # --- Power Iteration algorithm ---
        for iteration in range(self.num_iterations):

            # Compute Av (left singular vector)
            w = self._compute_matvec(self.A, v)
            w_norm = np.linalg.norm(w)
            if w_norm < self.tol:
                print(f"Converged at iteration {iteration}")
                break
            w /= w_norm

            # --- Compute A^T w (right singular vector) ---
            v_next = self._compute_matvec(self.A.T, w)
            v_next_norm = np.linalg.norm(v_next)
            if v_next_norm < self.tol or np.linalg.norm(v_next / v_next_norm - v) < self.tol:
                v = v_next / v_next_norm
                print(f"Converged at iteration {iteration}")
                break

            v = v_next / v_next_norm


        # --- Approximate the spectral norm by ||A @ v_k||_2 ---
        singular_value_est = np.linalg.norm(self._compute_matvec(self.A, v))

        return singular_value_est

def main():
    start_time = time.time()

    # --- Load the matrix A ---
    A = np.array([
        [3, 1, 0],
        [1, 2, 1],
        [0, 1, 3]
    ])
    num_iterations = 1000000
    tol = 1e-6

    # --- Perform the power iteration ---
    power_solver = PowerIteration(A, num_iterations, tol)
    dominant_eigenvalue = power_solver.solve()
    print("Dominant eigenvalue:", dominant_eigenvalue)
    with open("lambda.txt", "w+") as file:
        file.write(f"{dominant_eigenvalue}\n")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    main()
