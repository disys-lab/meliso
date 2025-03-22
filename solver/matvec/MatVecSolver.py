"""
@author: Paritosh Ramanan, Huynh Quang Nguyen Vo
MatVecSolver Module

This module implements an MPI-based distributed matrix-vector multiplication (MVM) solver. It abstracts 
the distributed computation between the root and non-root processes.
1. The root process (last rank) is responsible for initializing the matrix and vector.
2. The non-root processes participate in the distributed computation as needed.

Classes:
    MatVecSolver: Provides a high-level interface for distributed MVM, including methods for 
                  performing the MVM, running benchmarks for MVM, acquiring MCA statistics, and 
                  initializing data (matrix and vector).

Usage Example:
    solver = MatVecSolver()
    solver.initialize_data(matrix, vector)
    solver.matvec_mul(correction=True)
    solver.acquire_mca_stats()
    solver.benchmark_matvec_mul(hardware_on=0, scaling_on=0, correction=True)
    solver.finalize()
"""

from mpi4py import MPI
from .Root import Root
from .NonRoot import NonRoot

class MatVecSolver:
    """MPI-based distributed matrix-vector multiplication (MVM) solver."""

    def __init__(self) -> None:
        """
        Initialize the MatVecSolver by instantiating either the Root or NonRoot solver object based 
        on the MPI rank. The last rank is designated as the root process.
        """
        # Instantiate the MPI Communicator.
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # Use the last rank as the root process.
        if self.rank == self.size - 1:
            self.solver_object = Root(self.comm)
        else:
            self.solver_object = NonRoot(self.comm)

    def is_root_process(self) -> bool:
        """Check if the current process is the root process."""
        return self.rank == self.size - 1
    
    def is_non_root_process(self) -> bool:
        """Check if the current process is a non-root process."""
        return self.rank != self.size - 1
    
    def initialize_data(self, matrix, vector) -> None:
        """
        Initialize the matrix and vector data on the root process.

        Args:
            matrix (Any): The matrix to be used in MVM.
            vector (Any): The input vector for the MVM.
        """
        if not self.is_root_process():
            return None
        else:        
            self._initialize_mat(matrix)
            self._initialize_vector(vector)

    def _initialize_mat(self, matrix) -> None:
        """
        Initialize the matrix.

        Args:
            matrix (Any): The matrix to initialize.
        """
        self.solver_object.initializeMat(matrix)

    def _initialize_vector(self, vector) -> None:
        """
        Initialize the input vector.

        Args:
            vector (Any): The input vector to initialize.
        """
        self.solver_object.initializeX(vector)

    def matvec_mul(self, correction: bool = False) -> None:
        """
        Perform the distributed matrix-vector multiplication (MVM) on MCA devices.

        Args:
            correction (bool): Whether to apply a correction to the result (default is False).
        """
        self.solver_object.parallelMatVec(correction=correction)

    def benchmark_matvec_mul(
        self, hardware_on: int = 1, scaling_on: int = 0, correction: bool = False
    ) -> None:
        """
        Perform a benchmarked distributed MVM on software.

        Args:
            hardware_on (int): Flag to enable hardware acceleration (default is 1).
            scaling_on (int): Flag to enable scaling (default is 0).
            correction (bool): Whether to apply a correction to the result.
        """
        self.solver_object.benchmarkMatVecParallel(hardware_on, scaling_on, correction=correction)

    def acquire_mca_stats(self) -> None:
        """
        Acquire and report statistics from individual MCA.
        """
        self.solver_object.acquireMCAStats()

    def finalize(self) -> None:
        """
        Finalize the solver and perform any necessary cleanup.
        """
        self.solver_object.finalize()
