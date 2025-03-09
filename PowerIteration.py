from mpi4py import MPI
import numpy as np
import os
from solver.matvec.MatVecSolver import MatVecSolver
from typing import List, Tuple
import time

class PowerIteration:
    """
    Power Iteration Method for computing the dominant eigenvalue.
    """
    RESULT_FILENAME = "y_mem_result.csv"

    def __init__(self, A: np.ndarray, x_init: np.ndarray, lam_init: float,
                 num_iterations: int, tol: float) -> None:
        self.A = A
        self.A_trans = self.A.T
        self.x = x_init
        self.lam = lam_init
        self.num_iterations = num_iterations
        self.tol = tol

        # --- Instantiate the MatVecSolver object. ---
        self.mv_solver = MatVecSolver()

        # --- Save transpose matrix for external systems if required. ---
        np.savetxt("A.T.csv", self.A_trans, delimiter=",")

    def _compute_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Helper method to compute matrix-vector product using the external solver.
        """
        self.mv_solver.solverObject.initializeMat(matrix)
        self.mv_solver.solverObject.initializeX(vector)
        self.mv_solver.matVec(correction=True)
        return np.loadtxt(self.RESULT_FILENAME, delimiter=",")

    def solve(self) -> float:
        """
        Perform the two-sided power iteration and return the approximate spectral norm.
        """
        # --- Normalize the initial vector ---
        v_k = self.x.astype(float)
        v_norm = np.linalg.norm(v_k)
        if v_norm == 0:
            raise ValueError("Initial vector x_init has zero norm, cannot normalize.")
        v_k /= v_norm

        for _ in range(self.num_iterations):
            # --- Iteratively approximate left singular vector ---
            # w_k = A @ v_k
            w_k = self._compute_matvec(self.A, v_k)
            norm_w = np.linalg.norm(w_k)
            if norm_w < self.tol:
                break
            w_k /= norm_w

            # --- Iteratively approximate right singular vector ---
            # v_{k+1} = A.T @ w_k = (A.T @ A) @ k
            v_next = self._compute_matvec(self.A.T, w_k)
            norm_v_next = np.linalg.norm(v_next)
            if norm_v_next < self.tol:
                break
            v_k = v_next / norm_v_next

        # --- Approximate the spectral norm by ||A @ v_k||_2 ---
        w_final = self._compute_matvec(self.A, v_k)
        spectral_norm_est = np.linalg.norm(w_final)

        return spectral_norm_est

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    if rank == size - 1:
        start_time = time.time()
        # Load or define matrix A, initial vector x_init, and initial eigenvalue lam_init.
        A = np.loadtxt("A.csv", delimiter=",")
        x_init = np.random.rand(A.shape[1])
        lam_init = 0.0
        num_iterations = 1000
        tol = 1e-6

        power_solver = PowerIteration(A, x_init, lam_init, num_iterations, tol)
        dominant_eigenvalue = power_solver.solve()
        print("Dominant eigenvalue:", dominant_eigenvalue)
        with open("lambda.txt", "w+") as file:
            file.write(f"{dominant_eigenvalue}")

        end_time = time.time()
        print(f"Elapsed time: {start_time - end_time}")
    else:
        # Worker processes perform their assigned matrix-vector operations.
        MatVecSolver().matVec(correction=True)

if __name__ == "__main__":
    main()
