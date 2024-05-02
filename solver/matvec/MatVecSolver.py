from mpi4py import MPI
import numpy as np
import os,sys

from .Root import Root
from .NonRoot import NonRoot

class MatVecSolver:
    def __init__(self,set_mat=True):

        self.x = None
        self.y = None

        if MPI.COMM_WORLD.Get_rank() == MPI.COMM_WORLD.Get_size() - 1:
            self.solverObject = Root(MPI.COMM_WORLD,set_mat)
        else:
            self.solverObject = NonRoot(MPI.COMM_WORLD,set_mat)

    def matVec(self):
        self.solverObject.parallelMatVec()

    def centralizedBenchmarkMatVec(self):
        self.solverObject.benchmarkMatVec()

    def parallelizedBenchmarkMatVec(self, hardwareOn=0, scalingOn=0):
        self.solverObject.benchmarkMatVecParallel(hardwareOn,scalingOn)

    def acquireMCAStats(self):
        self.solverObject.acquireMCAStats()

    def finalize(self):
        self.solverObject.finalize()