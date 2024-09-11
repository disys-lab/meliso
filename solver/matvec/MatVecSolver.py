from mpi4py import MPI
import numpy as np
import os,sys

from .Root import Root
from .NonRoot import NonRoot

class MatVecSolver:
    def __init__(self,xvec= None):

        if MPI.COMM_WORLD.Get_rank() == MPI.COMM_WORLD.Get_size() - 1:
            self.solverObject = Root(MPI.COMM_WORLD)
            if xvec is None:
                xpath = os.environ["XVEC_PATH"]
                if xpath is None:
                    xpath = "inputs/vectors/input_x.txt"
                xvec = np.loadtxt(fname=xpath, delimiter=',')
            self.solverObject.initializeX(xvec)
            print("Successfully initialize vector x")

        else:
            self.solverObject = NonRoot(MPI.COMM_WORLD)

    def matVec(self,correction=False):
        print("Running VMM Operation...")
        self.solverObject.parallelMatVec(correction=correction)

    def centralizedBenchmarkMatVec(self):
        print("Running Centralized VMM Operation...")
        self.solverObject.benchmarkMatVec()

    def parallelizedBenchmarkMatVec(self, hardwareOn=0, scalingOn=0,correction=False):
        print("Running Parallelized VMM Operation...")
        self.solverObject.benchmarkMatVecParallel(hardwareOn,scalingOn,correction=correction)

    def acquireMCAStats(self):
        print("Acquring MCA Statistics...")
        self.solverObject.acquireMCAStats()

    def finalize(self):
        self.solverObject.finalize()