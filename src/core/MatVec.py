from .RootMCA import RootMCA
from .NonRootMCA import NonRootMCA
from mpi4py import MPI
import numpy as np

class MatVec:
    def __init__(self):

        # initialize MPI
        self.comm = MPI.COMM_WORLD
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()
        self.x = None
        self.y = None
        self.y_mem_result = None
        self.y_benchmark_result = None
        self.error = None
        if self.rank == self.comm.size-1:
            self.mca = RootMCA(self.comm)

        else:
            self.mca = NonRootMCA(self.comm)

    def parallelMatVec(self):
        if self.rank == self.comm.size - 1: #root process
            x = np.loadtxt(fname='input_x', delimiter=',')
            self.mca.x = x.reshape(x.shape[0],1)[:self.mca.matRows]

        self.y = self.mca.parallelMatVec()
        
        if self.rank == self.comm.size - 1:
            decomp_dir = self.mca.getDecompositionDir()
            self.y_mem_result = self.y[:self.mca.origMatRows]
            print(self.y_mem_result)

            # real_Ax = np.dot(self.mca.mat, self.mca.x)
            # print("y_rescaled:", y_rescaled_mem_result.reshape((1, self.mca.origMatRows)))
            # print("real_Ax:", real_Ax.reshape((1, self.mca.origMatRows)))
            # print(y_rescaled_mem_result.reshape((1, self.mca.origMatRows)) - real_Ax.reshape((1, self.mca.origMatRows)))

    def benchmarkMatVec(self):
        if self.rank == self.comm.size - 1:
            y = np.dot(self.mca.mat, self.mca.x)
            self.y_benchmark_result = y[:self.mca.origMatRows]
            self.error = self.y_mem_result - self.y_benchmark_result
            print("error", self.error)

    def benchmarkMatVecParallel(self,hardwareOn=0,scalingOn=0):
        # if self.rank == self.comm.size - 1:
        #     x = np.loadtxt(fname='input_x', delimiter=',')
        #     self.mca.x = x.reshape(x.shape[0],1)[:self.mca.matRows]
        # else:
        #     self.mca.meliso_obj.setHardwareOn(hardwareOn)
        #     self.mca.meliso_obj.setScalingOn(scalingOn)
        if self.rank != self.comm.size - 1:
            self.mca.meliso_obj.setHardwareOn(hardwareOn)
            self.mca.meliso_obj.setScalingOn(scalingOn)
            self.mca.initializeMCA()

        self.y = self.mca.parallelMatVec()

        if self.rank == self.comm.size - 1:
            decomp_dir = self.mca.getDecompositionDir()
            self.y_benchmark_result = self.y[:self.mca.origMatRows]
            print(self.y_benchmark_result)
            self.error = self.y_mem_result - self.y_benchmark_result
            print("error",self.error)
    
    def acquireMCAStats(self):
        self.mca.getMCAStats()
        if self.rank == self.comm.size-1:
            print("AllMCAStats\n")
            print(self.mca.allMCAStats)

    