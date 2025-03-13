import os
import time
from mpi4py import MPI
import numpy as np
from scipy.io import mmread, mmwrite
from solver.matvec.MatVecSolver import MatVecSolver
from typing import List, Tuple

class PDHGSolver:
    """
    @author: vohuynhquangnguyen
    Primal-Dual Hybrid Gradient (PDHG) solver for linear problems.
    """
    RESULT_FILENAME = "y_mem_result.csv"
    X_ITERATES_FILENAME = "x_iterates.csv"
    LOG_FILENAME = "x_log.txt"

    def __init__(self, A: np.ndarray, b: np.ndarray, c: np.ndarray, num_iterations: int) -> None:        
        """
        Args:
          A             : Constraint matrix for the problem.
          b             : RHS vector for the problem.
          c             : Objecive cost for the problem.
          num_iterations: Number of PDHG iterations.
        """
        self.A = A
        self.b = b
        self.c = c

        self.n_primal = A.shape[1]
        self.n_dual = A.shape[0]
        
        self.tol = 1e-6
        self.theta = 1.0
        self.num_iterations = num_iterations

        self.x_iterates: List[np.ndarray] = []
        self.x_bar = None
        self.x_avg = None

        # --- Instantiate the MatVecSolver object. ---
        self.mv_solver = MatVecSolver()
        
    # --- Matrix-Vector Multiplication (MVM) ---
    def _compute_matvec(self, matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
        """
        Helper method to compute matrix-vector product using external solver.
        """
        self.mv_solver.solverObject.initializeMat(matrix)
        self.mv_solver.solverObject.initializeX(vector)
        self.mv_solver.matVec(correction=True)
        return np.loadtxt(self.RESULT_FILENAME, delimiter=",")
    
    # --- 
    @staticmethod
    def _project_dual(mu: np.ndarray) -> np.ndarray:
        """
        Helper method to project dual variables to be non-negative.
        """
        return np.maximum(mu, 0)
    
    @staticmethod
    def _project_primal(x: np.ndarray) -> np.ndarray:
        """
        Helper method to project primal variables to be non-negative.
        """
        return np.maximum(x, 0)

    @staticmethod
    def _compute_stepsize(A):
        """
        Helper method to compute primal and dual steps.
        """
        maximum_step = 1 / np.linalg.norm(A, ord = 2)
        primal_step = maximum_step
        dual_step = maximum_step
        return primal_step, dual_step   

    def solve(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Run the PDHG iterations and return final and averaged solutions.
        """
        
        # --- Remove existing log file (if any) before starting the iterations ---
        if os.path.exists(self.LOG_FILENAME):
            os.remove(self.LOG_FILENAME)

        # --- Compute step size ---
        primal_step, dual_step = self._compute_stepsize(self.A)

        # --- Initialize x and mu ---
        x = np.zeros(self.n_primal)
        mu = np.zeros(self.n_dual)
        self.x_bar = x.copy()
    
        for k in range(self.num_iterations):

            # --- Dual update using extrapolated x_bar ---
            # Set the accelerator's matrix to A for computing A * x_bar
            mv_result = self._compute_matvec(self.A, x_bar)
            mu_tilde = mu + dual_step * (mv_result - self.b)
            mu_next = self._project_dual(mu_tilde)

            # --- Primal update ---
            # Set the accelerator's matrix to A for computing A.T @ mu_next
            mv_result = self._compute_matvec(self.A.T, mu_next)
            x_grad = x - primal_step * (mv_result + self.c)
            x_next = self._project_primal(x_grad)

            # --- Extrapolation for next x_bar ---
            x_bar = x_next + self.theta * (x_next - x)

            # --- Check convergence (primal residual) ---
            if np.linalg.norm(x_next - x) < self.tol:
                print(f"Terminated after {k} iterations.")
                break
            
            # --- Update primals and duals for next iteration.
            self.x_iterates.append(x_next.copy())
            x = x_next
            mu = mu_next

            # --- Append current iterate of x to the log file ---
            with open(self.LOG_FILENAME, "a+") as file:
                file.write(f"{self.x}\n")
                
        # --- Remove previous all-iterates file (if any) before saving ---
        if os.path.exists(self.X_ITERATES_FILENAME):
            os.remove(self.X_ITERATES_FILENAME)        
        np.savetxt(self.X_ITERATES_FILENAME, np.array(self.x_iterates), delimiter=",")
        self.x_avg = np.mean(self.x_iterates, axis=0)

        return self.x_bar, self.x_avg
    
def main() -> None:
    """Main execution block with MPI context management."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    root_rank = size - 1
    
    # --- The PDHG algorithm is executed only on the designated Root process ---
    if rank == root_rank:
        start_time = time.time()

        # Load input data from environment variables
        A_file = os.environ["A_FILE"]
        b_file = os.environ["B_FILE"]
        c_file = os.environ["C_FILE"]

        A = np.loadtxt(A_file, delimiter=",")
        b = np.loadtxt(b_file, delimiter=",")
        c = np.loadtxt(c_file, delimiter=",")

        # Configure solver parameters
        solver = PDHGSolver(A=A, b=b, c=c, num_iterations=100000)

        # Execute optimization
        x_final, _ = solver.solve()

        print(f"Optimal solution: {x_final}")
        print(f"Objective value: {-np.dot(c, x_final):.2f}")

        end_time = time.time()
        print(f"Elapsed time: {end_time - start_time}")

        # Broadcast termination signal to all workers
        comm.bcast(True, root=root_rank)

    else:
        # Worker processes loop until termination signal is received
        while True:
            # Wait for broadcast from main process
            done = comm.bcast(None, root=size-1)
            if done:
                break
            MatVecSolver().matVec(correction=True)

if __name__ == "__main__":
    main()