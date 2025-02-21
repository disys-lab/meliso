from mpi4py import MPI
import numpy as np
from scipy.io import mmread, mmwrite
from solver.matvec.MatVecSolver import MatVecSolver

class PDHGSolver:
    def __init__(self, A_file, x_init, mu_init, b, c, K, eta1, eta2):
        """
        PDHG solver constructor.
        
        Parameters:
          A_file  : str
                    Path to the input matrix (CSV format).  
          x_init  : np.ndarray
                    Initial x vector (primal variable).
          mu_init : np.ndarray
                    Initial mu vector (dual variable).
          b, c    : np.ndarray
                    Bias vectors (with appropriate dimensions).
          K       : int
                    Number of PDHG iterations.
          eta1    : float
                    Step size for x-update.
          eta2    : float
                    Step size for mu-update.
        """
        # Load matrix A and compute/store its transpose.
        self.A = np.loadtxt(A_file, delimiter=",")
        self.A_dim = self.A.shape  # e.g., (m, n)
        self.A_trans = self.A.T
        # Save A_trans for record if needed.
        np.savetxt("A.T.csv", self.A_trans, delimiter=",")
        self.A_trans_dim = self.A_trans.shape  # (n, m)
        
        # Store PDHG variables and parameters.
        self.x = x_init
        self.mu = mu_init
        self.b = b
        self.c = c
        self.K = K
        self.eta1 = eta1
        self.eta2 = eta2
        self.x_iterates = []
        
        # Instantiate the distributed accelerator.
        self.mv_solver = MatVecSolver()

    def solve(self):
        """
        Run the PDHG iterations. For the x-update, we require Aᵀ*mu.
        To do that we instruct the accelerator to use A_trans.
        For the mu-update, we switch back to A.
        """
        for k in range(self.K):
            # --- x-update: x = max(x - eta1*(Aᵀ*mu + c), 0) ---
            # Set the accelerator’s matrix to A_trans for computing Aᵀ * mu.
            self.mv_solver.solverObject.initializeMat(self.A_trans)
            A_trans_mu = self.mv_solver.matVec(self.mu)
            x_tilde = self.x - self.eta1 * (A_trans_mu + self.c)
            x_next = np.maximum(x_tilde, 0)
            
            # --- mu-update: mu = max(mu + eta2*(A*(2*x_next - x) - b), 0) ---
            temp_vec = 2 * x_next - self.x
            self.mv_solver.solverObject.initializeMat(self.A)  # switch back to A
            A_temp = self.mv_solver.matVec(temp_vec)
            self.mu = np.maximum(self.mu + self.eta2 * (A_temp - self.b), 0)
            
            self.x_iterates.append(x_next.copy())
            self.x = x_next  # update for next iteration
        
        x_last = self.x
        x_avg = np.mean(self.x_iterates, axis=0)
        return x_last, x_avg

def main():
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()
    
    # The PDHG algorithm is executed only on the designated Root process.
    # According to your setup, we assume the Root process is rank = size - 1.
    if rank == size - 1:
        # Path to matrix A.
        A_file = "../A.csv"
        # Load A to determine its dimensions.
        A = np.loadtxt(A_file, delimiter=",")
        m, n = A.shape  # A is (m x n): mu has dimension (m,1) and x (n,1)
        
        # Path to bias vectors b and c.
        b_file = "../b.csv"
        c_file = "../c.csv"
        b = np.loadtxt(b_file, delimiter=",")
        c = np.loadtxt(c_file, delimiter=",")

        # Initialize variables.
        x_init = np.random.rand(n, 1)
        mu_init = np.random.rand(m, 1)
        
        # PDHG parameters.
        K = 100      # number of iterations
        eta1 = 0.01  # step size for x-update
        eta2 = 0.01  # step size for mu-update
        
        # Create and run the PDHG solver.
        solver = PDHGSolver(A_file, x_init, mu_init, b, c, K, eta1, eta2)
        x_last, x_avg = solver.solve()
        
        print("Last iterate x_K:")
        print(x_last)
        print("\nAverage of iterates:")
        print(x_avg)
    else:
        # Non-root processes run the accelerator service.
        mv_solver = MatVecSolver()
        mv_solver.matVec(correction=False)

if __name__ == "__main__":
    main()
