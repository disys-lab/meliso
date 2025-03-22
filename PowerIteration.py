import os
import time
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver
from typing import List, Tuple
from scipy.io import mmread

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

        # --- Save transpose matrix for external systems if required. ---
        np.savetxt("A.T.csv", self.A.T, delimiter=",")

    @staticmethod
    def _normalized_vector(x: np.ndarray) -> Tuple[float, np.ndarray]:
        """
        Helper method to normalize a vector.
        """
        norm_x = np.linalg.norm(x)
        x_normalized = x / norm_x if norm_x > 0 else x
        return norm_x, x_normalized
    
    def _compute_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Helper method to compute matrix-vector product using the MELISO matvec solver.
        """
        self.mv_solver.initializeMat(matrix)
        self.mv_solver.initializeX(vector)
        self.mv_solver.matVec(correction=True)
        self.mv_solver.finalize()
        self.mv_solver.acquireMCAStats()
        return np.loadtxt(self.RESULT_FILENAME, delimiter=",")
    
    def solve(self) -> float:
        """
        Perform the two-sided power iteration and return the approximate spectral norm.
        """
        # --- Normalize the initial vector ---
        v = np.random.rand(self.A.shape[1])
        _ , v = self._normalized_vector(v)
        last_v = v.copy()

        for i in range(self.num_iterations):
            # --- Iteratively approximate left singular vector (w_k = A @ v_k) ---
            w = self._compute_matvec(self.A, v)
            norm_w, w = self._normalized_vector(w)
            if norm_w < self.tol:
                print(f"Iteration {i}: Convergence reached with norm_w = {norm_w}")
                break

            # --- Iteratively approximate right singular vector (v_{k+1} = A.T @ w_k = (A.T @ A) @ k) ---
            v_next = self._compute_matvec(self.A.T, w)
            norm_v_next, v_next = self._normalized_vector(v_next)
            if norm_v_next < self.tol:
                print(f"Iteration {i}: Convergence reached with norm_v_next = {norm_v_next}")
                break

            last_v = v_next.copy()  # Update last valid vector
            v = v_next.copy()

        # --- Approximate the spectral norm by ||A @ v_k||_2 ---
        w_final = self._compute_matvec(self.A, last_v)
        spectral_norm_est = np.linalg.norm(w_final)

        return spectral_norm_est

def main():
    start_time = time.time()

    # --- Load the matrix A ---
    A = mmread(os.getenv("A_FILE", "inputs/matrices/A.mtx")).toarray()
    num_iterations = 1000
    tol = 1e-6

    # --- Perform the power iteration ---
    power_solver = PowerIteration(A, num_iterations, tol)
    dominant_eigenvalue = power_solver.solve()
    print("Dominant eigenvalue:", dominant_eigenvalue)
    with open("lambda.txt", "w+") as file:
        file.write(f"{dominant_eigenvalue}")

    end_time = time.time()
    print(f"Elapsed time: {end_time - start_time}")

if __name__ == "__main__":
    main()
