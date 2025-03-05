from mpi4py import MPI
import numpy as np
import os
from typing import Optional

from .Root import Root
from .NonRoot import NonRoot

class MatVecSolver:
    """MPI-based distributed matrix-vector multiplication (MVM) solver."""

    def __init__(self, xvec: Optional[np.ndarray] = None) -> None:
        """
        Args:
            xvec    : (Optional) Input vector
        """

        # --- Instantiate the MPI Communicator ---
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        size = comm.Get_size()

        # --- Use the last rank as the root process ---
        if rank == size - 1:
            self.solverObject = Root(comm)

            # In the root process, we load the input vector
            if xvec is None:
                # Safely get the path from the environment; fallback to default if not set.
                xpath = os.getenv("XVEC_PATH", "inputs/vectors/input_x.txt")
                xvec = np.loadtxt(fname=xpath, delimiter=',')
            self.solverObject.initializeX(xvec)

        else:
            # --- Use other ranks as the non-root processes
            self.solverObject = NonRoot(comm)

    def matVec(self, correction: bool = False) -> None:
        """Perform the distributed MVM."""
        self.solverObject.parallelMatVec(correction=correction)

    def parallelizedBenchmarkMatVec(self, hardwareOn: int = 1, scalingOn: int = 0, 
                                    correction = False) -> None:
        """
        Perform the benchmarked distributed MVM.

        Args:
            hardware_on     : Flag to enable hardware acceleration (default is 1).
            scaling_on      : Flag to enable scaling (default is 0).
            correction      : Whether to apply a correction to the result.
        """
        self.solverObject.benchmarkMatVecParallel(hardwareOn,scalingOn,correction=correction)

    def acquireMCAStats(self) -> None:
        """Acquire and report Memristor Crossbar Array (MCA) statistics."""        
        self.solverObject.acquireMCAStats()

    def finalize(self):
        self.solverObject.finalize()