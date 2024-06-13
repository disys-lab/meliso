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
                #obtain x here
                # xpath = "input_x.txt"
                xpath = os.environ["XVEC_PATH"]
                xvec = np.loadtxt(fname=xpath, delimiter=',')

            # you can set a raw unprocessed matrix here or have the RootMCA read directly from config file
            # for instance you can do:
            # mat = np.random.rand(128,128)
            # self.solverObject = Root(MPI.COMM_WORLD,x=xvec,mat=mat)

            #to reinitialize the matrix you can do for instance:
            #self.solverObject.initializeMat(np.random.rand(128,128))
            self.solverObject.initializeX(xvec)

        else:
            self.solverObject = NonRoot(MPI.COMM_WORLD)

    def matVec(self,correction=False):
        self.solverObject.parallelMatVec(correction=correction)

    def centralizedBenchmarkMatVec(self):
        self.solverObject.benchmarkMatVec()

    def parallelizedBenchmarkMatVec(self, hardwareOn=0, scalingOn=0,correction=False):
        self.solverObject.benchmarkMatVecParallel(hardwareOn,scalingOn,correction=correction)

    def acquireMCAStats(self):
        self.solverObject.acquireMCAStats()

    def finalize(self):
        self.solverObject.finalize()