from mpi4py import MPI
import numpy as np
import os
from solver.matvec.MatVecSolver import MatVecSolver
from typing import List, Tuple

class PowerIteration:
    """
    Power Iteration Method for computing the dominant eigenvalue.
    """
    RESULT_FILENAME = "y_mem_result.csv"

    def __init__(self, A: np.ndarray, x_init: np.ndarray, lam_init: float,
                 num_iterations: int, tol: float) -> None:
        self.A = A
        self.x = x_init
        self.lam = lam_init
        self.num_iterations = num_iterations
        self.tol = tol
        # Instantiate the MatVecSolver as in the PDHG reference code.
        self.mv_solver = MatVecSolver()

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
        Perform the power iteration method and return the dominant eigenvalue.
        """
        for _ in range(self.num_iterations):
            # Compute y = A * x
            y = self._compute_matvec(self.A, self.x)
            # Compute the norm of y and check for zero to avoid division errors.
            norm_y = np.linalg.norm(y)
            if norm_y == 0:
                raise ValueError("The computed vector has zero norm; cannot normalize.")
            # Normalize y to obtain the new eigenvector approximation.
            x_new = y / norm_y
            # Compute the new eigenvalue using the Rayleigh quotient:
            # lam_new = (x_new.T @ (A * x_new)) / (x_new.T @ x_new)
            Ax_new = self._compute_matvec(self.A, x_new)
            lam_new = (x_new.T @ Ax_new) / (x_new.T @ x_new)
            # Check for convergence.
            if np.abs(lam_new - self.lam) < self.tol:
                self.lam = lam_new
                self.x = x_new
                break
            # Update eigenvalue and eigenvector for the next iteration.
            self.lam = lam_new
            self.x = x_new
        return self.lam

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    # Assume that only the designated process (say, the last one) runs the iteration.
    if rank == size - 1:
        # Load or define matrix A, initial vector x_init, and initial eigenvalue lam_init.
        A = np.loadtxt("A.csv", delimiter=",")
        x_init = np.ones(A.shape[1])  # or any nonzero starting vector
        lam_init = 0.0
        num_iterations = 1000
        tol = 1e-6

        power_solver = PowerIteration(A, x_init, lam_init, num_iterations, tol)
        dominant_eigenvalue = power_solver.solve()
        print("Dominant eigenvalue:", dominant_eigenvalue)
        with open("lambda.txt", "w+") as file:
            file.write(f"{dominant_eigenvalue}")
    else:
        # Worker processes perform their assigned matrix-vector operations.
        MatVecSolver().matVec(correction=True)

if __name__ == "__main__":
    main()
