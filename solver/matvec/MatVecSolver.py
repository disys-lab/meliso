# ==================================================================================================
# @author: Huynh Quang Nguyen Vo, Paritosh Ramanan
# @affiliation: Oklahoma State University
# @date: 2026-02-24
# ==================================================================================================
import os
import numpy as np

from mpi4py import MPI
from .Root import Root
from .NonRoot import NonRoot

class MatVecSolver:
    """
    `MatVecSolver` serves as the main interface for performing distributed matrix-vector multiplication 
    (MVM) using MPI. The interface abstracts away the complexities of parallelization and allows 
    users to easily execute MVM.

    A simple distributed MVM includes:
    1. Initializing the input vector and matrix (if applicable).
    2. Performing the MVM operation in parallel across multiple processes.
    3. Optionally applying min-max scaling reversion to the results.
    4. Finalizing the MPI environment after computations are complete.
    5. Acquiring MCA (Memristor Crossbar Array) statistics for performance analysis.
    """
    def __init__(self, xvec= None, mat = None):
        """
        Initializes the MatVecSolver instance.
        Sets up the input vector and matrix, initializes MPI communication, 
        and assigns the appropriate solver object (Root or NonRoot) based on the MPI rank.
        Args:
            xvec (numpy.ndarray, optional): The input vector. If not provided, it is loaded 
                from the file path specified in the 'XVEC_PATH' environment variable, 
                defaulting to 'inputs/vectors/input_x.txt'. Defaults to None.
            mat (numpy.ndarray, optional): The input matrix. Defaults to None.
        """
        if xvec is None:
            print("No input vector provided, loading from default path...\n")
            xpath = os.environ.get("XVEC_PATH", "inputs/vectors/input_x.txt")
            xvec = np.loadtxt(fname=xpath, delimiter=',')

        self.xvec = xvec
        self.mat = mat
        self.y = None
        
        # MPI Handler
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        if self.rank == self.size - 1:
            self.solverObject = Root(MPI.COMM_WORLD, x=self.xvec, mat=self.mat)
        else:
            self.solverObject = NonRoot(MPI.COMM_WORLD)

    # ----------------------------------------------------------------------------------------------
    # Explicit initializers for the input vector and matrix
    # ----------------------------------------------------------------------------------------------
    def initializeVec(self):
        """
        Initializes the vector using the underlying solver object.

        If the `solverObject` attribute is an instance of `Root`, this method delegates the 
        initialization by calling its `initializeX` method with the current vector (`self.xvec`).
        """
        if isinstance(self.solverObject, Root):
            self.solverObject.initializeX(self.xvec)

    def initializeMat(self):
        """
        Initializes the matrix using the underlying solver object.

        If the `solverObject` attribute is an instance of `Root`, this method delegates the 
        initialization by calling its `initializeMat` method with the current matrix (`self.mat`).
        """
        if isinstance(self.solverObject, Root):
            self.solverObject.initializeMat(self.mat)

    # ----------------------------------------------------------------------------------------------
    # Core operations 
    # ----------------------------------------------------------------------------------------------
    def matVec(self, correction=False):
        """
        Performs a parallel MVM using the underlying solver object.

        Args:
            correction (bool, optional): A flag indicating whether to apply a min-max scaling 
            reversion during the MVM. Defaults to False.
        """
        self.solverObject.parallelMatVec(correction=correction)

    def parallelizedBenchmarkMatVec(self, hardwareOn=1, scalingOn=0, correction=False):
        """
        Executes a parallelized benchmark for MVM.

        This method delegates the benchmarking execution to the underlying solver object.

        Args:
            hardwareOn (int, optional): Flag to enable or disable hardware execution. Defaults to 1.
            scalingOn (int, optional): Flag to enable or disable scaling. Defaults to 0.
            correction (bool, optional): Flag to apply correction during the benchmark. Defaults to False.
        """
        self.solverObject.benchmarkMatVecParallel(hardwareOn, scalingOn, correction=correction)

    def acquireMCAStats(self):
        """
        Acquires the MCA (memristor crossbar array) statistics from the underlying solver object.

        This method delegates the operation to the internal `solverObject` to gather and process the 
        relevant statistical data (mean/stddev/maximum/minimum read and write energy/latency).
        """
        self.solverObject.acquireMCAStats()

    def finalize(self):
        """
        Finalizes the solver.

        This method delegates the finalization process to the underlying solver object, ensuring that 
        any necessary cleanup, memory deallocation, or termination procedures are properly executed.
        """
        self.solverObject.finalize()

    def acquireResults(self):
        """
        Acquires and returns the results from the solver object.

        If the `solverObject` attribute is an instance of `Root`, this method retrieves the MVM result 
        (`y_mem_result`) from the solver object, assigns it to `self.y`, and returns it.

        Returns:
            Any: The result data (`y_mem_result`) from the solver object if it is an instance of 
            `Root`, otherwise `None`.
        """
        if isinstance(self.solverObject, Root):
            self.y = self.solverObject.y_mem_result
            return self.y