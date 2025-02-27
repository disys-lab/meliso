from mpi4py import MPI
import numpy as np
import os
from scipy.io import mmread, mmwrite
from solver.matvec.MatVecSolver import MatVecSolver
from typing import List, Tuple

class PDHGSolver:
    """
    @author: vohuynhquangnguyen
    Primal-Dual Hybrid Gradient (PDHG) solver for the problem:
        min_x max_mu {<A*x, mu> - <b, mu> - <c, x> : x >= 0, mu >= 0}. 
    """
    RESULT_FILENAME = "y_mem_result.csv"
    X_ITERATES_FILENAME = "x_iterates.csv"

    def __init__(self, A: np.ndarray, x_init: np.ndarray, mu_init: np.ndarray, b: np.ndarray, c: np.ndarray,
                 num_iterations: int, primal_step: float, dual_step: float) -> None:        
        """
        Parameters:
          A            : Input matrix for the problem.
          x_init       : Initial primal variable.
          mu_init      : Initial dual variable.
          b, c         : Bias vectors.
          num_iterations: Number of PDHG iterations.
          primal_step  : Step size for primal updates.
          dual_step    : Step size for dual updates.
        """
        self.A = A
        self.A_trans = self.A.T
        self.x = x_init
        self.mu = mu_init
        self.b = b
        self.c = c
        self.num_iterations = num_iterations
        self.primal_step = primal_step
        self.dual_step = dual_step
        self.x_iterates: List[np.ndarray] = []

        # Instantiate the MatVecSolver object.
        self.mv_solver = MatVecSolver()
        
        # Save transpose matrix for external systems if required.
        np.savetxt("A.T.csv", self.A_trans, delimiter=",")

    def _compute_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """Helper method to compute matrix-vector product using external solver."""
        self.mv_solver.solverObject.initializeMat(matrix)
        self.mv_solver.matVec(vector)
        return np.loadtxt(self.RESULT_FILENAME, delimiter=",")
    
    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """Run the PDHG iterations and return final and averaged solutions."""

        for _ in range(self.num_iterations):
            # --- Primal update: x_next = max(x - eta_p*(A.T*mu + c), 0) ---
            # Set the accelerator's matrix to A.T for computing A.T * mu.
            A_trans_mu = self._compute_matvec(self.A_trans, self.mu)

            # Update x using the PDHG update rule.
            x_tilde = self.x - self.primal_step * (A_trans_mu + self.c)
            x_next = np.maximum(x_tilde, 0)
            
            # --- Dual-update: mu = max(mu + eta_d*(A*(2*x_next - x) - b), 0) ---
            # Set the accelerator's matrix to A for computing A * (2*x_next - x).
            temp_vec = 2 * x_next - self.x
            A_temp = self._compute_matvec(self.A, temp_vec)

            # Update mu using the PDHG update rule.
            self.mu = np.maximum(self.mu + self.dual_step * (A_temp - self.b), 0)
            self.x_iterates.append(x_next.copy())
            self.x = x_next
            
        # --- Save all iterates once after optimization completes ---
        np.savetxt(self.X_ITERATES_FILENAME, np.array(self.x_iterates), delimiter=",")
        x_avg = np.mean(self.x_iterates, axis=0)
        return self.x, x_avg

def main() -> None:
    """Main execution block with MPI context management."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root_rank = size - 1
    
    # --- The PDHG algorithm is executed only on the designated Root process ---
    if rank == root_rank:
        # Load input data from environment variables
        A_file = os.environ["A_FILE"]
        b_file = os.environ["B_FILE"]
        c_file = os.environ["C_FILE"]

        A = np.loadtxt(A_file, delimiter=",")
        b = np.loadtxt(b_file, delimiter=",")
        c = np.loadtxt(c_file, delimiter=",")

        # Initialize variables based on matrix dimensions
        x_init = np.zeros(A.shape[1])
        mu_init = np.zeros(A.shape[0])

        # Configure solver parameters
        solver = PDHGSolver(A=A, x_init=x_init,mu_init=mu_init,b=b,c=c,
                            num_iterations=3000,primal_step=0.16,dual_step=0.16)

        # Execute optimization
        x_last, x_avg = solver.solve()

        print(f"Last iterate x_K: {x_last}")
        print(f"Average of iterates: {x_avg}")

    else:
        # Worker processes handle matrix-vector operations
        MatVecSolver().matVec(correction=False)

if __name__ == "__main__":
    main()