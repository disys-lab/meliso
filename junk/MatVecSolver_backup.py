from mpi4py import MPI
import numpy as np
import os,sys

from ..solver.matvec.Root import Root
from ..solver.matvec.NonRoot import NonRoot

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
        if isinstance(self.solverObject, Root):
            self.solverObject.initializeX(self.xvec)

    def initializeMat(self):
        if isinstance(self.solverObject, Root):
            self.solverObject.initializeMat(self.mat)

    # ----------------------------------------------------------------------------------------------
    # Core operations 
    # ----------------------------------------------------------------------------------------------
    def matVec(self,correction=False):
        self.solverObject.parallelMatVec(correction=correction)

    def parallelizedBenchmarkMatVec(self, hardwareOn=1, scalingOn=0,correction=False):
        self.solverObject.benchmarkMatVecParallel(hardwareOn,scalingOn,correction=correction)

    def acquireMCAStats(self):
        self.solverObject.acquireMCAStats()

    def finalize(self):
        self.solverObject.finalize()

    def acquireResults(self):
        if isinstance(self.solverObject, Root):
            self.y = self.solverObject.y_mem_result
            return self.y