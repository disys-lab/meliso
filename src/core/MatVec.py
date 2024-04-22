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
        if self.rank == self.comm.size-1:
            self.mca = RootMCA(self.comm)

        else:
            self.mca = NonRootMCA(self.comm)
    
    def parallelMatVec(self):
        if self.rank == self.comm.size - 1: #root process
            self.mca.x = np.random.randint(0, 10000, size=(self.mca.matRows, 1)) / 10000.0

        self.y = self.mca.parallelMatVec()
        
        if self.rank == self.comm.size - 1:
            decomp_dir = self.mca.getDecompositionDir()
            y_result = self.y[self.mca.origMatRows]
            np.savetxt(y_result,"{}/y_result.txt".format(decomp_dir))
            return self.y[self.mca.origMatRows]
    
    def acquireMCAStats(self):
        self.mca.getMCAStats()
        if self.rank == self.comm.size-1:
            print("AllMCAStats\n")
            print(self.mca.allMCAStats)

    