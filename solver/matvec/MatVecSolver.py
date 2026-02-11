from mpi4py import MPI
import numpy as np
import os,sys

from .Root import Root
from .NonRoot import NonRoot

class MatVecSolver:
    def __init__(self, xvec= None, mat = None):
        if xvec is None:
            print("No input vector provided, loading from default path...\n")
            xpath = os.environ.get("XVEC_PATH", "inputs/vectors/input_x.txt")
            xvec = np.loadtxt(fname=xpath, delimiter=',')

        self.xvec = xvec
        self.mat = mat
        
        # Use rank 0 for the root process (standard convention)
        if MPI.COMM_WORLD.Get_rank() == MPI.COMM_WORLD.Get_size() - 1:
            self.solverObject = Root(MPI.COMM_WORLD)
        else:
            self.solverObject = NonRoot(MPI.COMM_WORLD)

    def initializeVec(self):
        if isinstance(self.solverObject, Root):
            self.solverObject.initializeX(self.xvec)

    def initializeMat(self):
        if isinstance(self.solverObject, Root):
            self.solverObject.initializeMat(self.mat)

    def matVec(self,correction=False):
        self.solverObject.parallelMatVec(correction=correction)

    def parallelizedBenchmarkMatVec(self, hardwareOn=1, scalingOn=0,correction=False):
        self.solverObject.benchmarkMatVecParallel(hardwareOn,scalingOn,correction=correction)

    def acquireMCAStats(self):
        self.solverObject.acquireMCAStats()

    def finalize(self):
        self.solverObject.finalize()