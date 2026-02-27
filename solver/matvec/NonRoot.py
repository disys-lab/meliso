# ==================================================================================================
# @author: Huynh Quang Nguyen Vo, Paritosh Ramanan
# @affiliation: Oklahoma State University
# @date: 2026-02-25
# ==================================================================================================

import os, sys
import numpy as np

#===================================================================================================
# Import MELISO+ core
#===================================================================================================
if "MELISO_SRC_PATH" in os.environ.keys():
    if not os.isdir(os.environ["MELISO_SRC_PATH"]):
        raise Exception("Env Var {} MELISO_SRC_PATH is invalid!".format(os.environ["MELISO_SRC_PATH"]))
    sys.path.append(os.environ["MELISO_SRC_PATH"])
else:
    sys.path.append("../../")
from src.core.NonRootMCA import NonRootMCA

class NonRoot:
    def __init__(self, comm, verbose=False):
        """
        Initialize a `NonRoot` solver instance.

        Args:
        comm: MPI communicator for distributed operations.
        verbose: Flag to enable verbose output for debugging purposes (default: False).

        Attributes:
        comm (MPI.Comm): Stores the MPI communicator for inter-process communication.
        virtualizationOn (bool): Flag to enable virtualization, set to True by default.
        mca (NonRootMCA): Instance of NonRootMCA initialized with the provided communicator without setting the matrix (set_mat=False).
        """
        self.comm = comm
        self.virtualizationOn = True
        self.mca = NonRootMCA(self.comm,set_mat=False)
        self.verbose = verbose

    def awaitInstructions(self):
        """
        Wait for and process broadcast instructions from the root process.
        This method blocks until the root process broadcasts a signal indicating
        whether non-root processes should proceed with matrix-vector multiplication
        (MVM) or exit. The signal is conveyed through a 2-element array where the
        first element determines the action.
        The broadcast array contains row and column indices of a submatrix:
        - If data[0] >= 0: Non-root processes should proceed with MVM by setting
          the matrix and returning True.
        - If data[0] < 0: Non-root processes should exit and return False.
        
        Returns:
        bool: True if non-root processes should proceed with MVM,
            False if they should exit.

        """
        data = np.array([-1, -1], dtype=np.float64)
        if self.verbose:
            print(f"[INFO] RANK{self.mca.rank}: trying to receive the next set of row and column indices of the submatrix")
            
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)
        if self.verbose:
            print(f"[INFO] RANK{self.mca.rank}: row and column indices of the submatrix {data}")

        # The root process will send a signal to all non-root processes to indicate whether they 
        # should proceed with the MVM or exit.
        if data[0] >= 0: # If the first element of the received data is non-negative, it indicates that the non-root processes should proceed with the MVM.
            self.mca.setMat()
            return True
        else: # If the first element of the received data is negative, it indicates that the non-root processes should exit.
            if self.verbose:
                print(f"[INFO] RANK{self.mca.rank}: has exited", data)
            return False

    def parallelMatVec(self, correction=False):
        """
        Perform parallel matrix-vector multiplication (MVM) using the `NonRootMCA` instance.
        This method is designed to be called by non-root processes in a distributed computing environment.
        """

        # Placeholder for the min-max scaling reversal logic, since both `Root` and `NonRoot` share 
        # the same `parallelMatVec` method.
        correction = correction

        if self.virtualizationOn: # If virtualization is enabled, the non-root processes will continuously wait for instructions from the root process to perform MVM until they receive a signal to exit.
            while self.awaitInstructions():
                self.y = self.mca.parallelMatVec()
            if self.verbose:
                print(f"[INFO] RANK{self.mca.rank}: leaving parallelMatVec")
        else:
            if self.verbose:
                print(f"[INFO] RANK{self.mca.rank}: virtualization is OFF")
            self.y = self.mca.parallelMatVec()

    def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0, correction= False):
        """
        Benchmark the parallel matrix-vector multiplication (MVM) process using the `NonRootMCA` instance.
        """
        if self.verbose:
            print(f"[INFO] RANK{self.mca.rank}: benchmarking started...")
        self.mca.meliso_obj.setHardwareOn(hardwareOn)
        self.mca.meliso_obj.setScalingOn(scalingOn)
        self.parallelMatVec(correction)

        if self.verbose:
            print(f"[INFO] RANK{self.mca.rank}: benchmarking complete")

    def acquireMCAStats(self):
        """
        Acquire and return the MCA statistics from the `NonRootMCA` instance.
        """
        self.mca.getMCAStats()

    def finalize(self):
        """
        Placeholder for any necessary cleanup operations before finalizing the non-root processes.
        """
        pass