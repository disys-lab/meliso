import os,sys
import numpy as np
if "MELISO_SRC_PATH" in os.environ.keys():
    if not os.isdir(os.environ["MELISO_SRC_PATH"]):
        raise Exception("Env Var {} MELISO_SRC_PATH is invalid!".format(os.environ["MELISO_SRC_PATH"]))
    sys.path.append(os.environ["MELISO_SRC_PATH"])
else:
    sys.path.append("../../")
from src.core.NonRootMCA import NonRootMCA

class NonRoot:
    def __init__(self,comm):
        self.comm = comm
        self.virtualizationOn = True
        self.mca = NonRootMCA(self.comm,set_mat=False)

    def awaitInstructions(self):
        data = np.array([-1, -1], dtype=np.float64)
        #print("RANK{}: trying to recieve next set of row and col id of submatrix".format(self.mca.rank))
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)
        #print("RANK{}: row and col id of submatrix".format(self.mca.rank), data)
        if data[0] >= 0:
            self.mca.setMat()
            return True
        else:
            #print("RANK{}: has exited".format(self.mca.rank), data)
            return False

    def parallelMatVec(self,correction=False):
        if self.virtualizationOn:
            while self.awaitInstructions():
                self.y = self.mca.parallelMatVec()
            #print("RANK{}: leaving parallelMatVec".format(self.mca.rank))
        else:
            #print("RANK{}: virtualizationON".format(self.virtualizationOn))
            self.y = self.mca.parallelMatVec()

        
        self.comm.gather([self.mca.errorCorrectionTime],root=self.ROOT_PROCESS_RANK)

    def benchmarkMatVec(self):
       pass

    def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0,correction= False):
        #print("RANK{}: benchmarking started".format(self.mca.rank))
        self.mca.meliso_obj.setHardwareOn(hardwareOn)
        self.mca.meliso_obj.setScalingOn(scalingOn)
        #self.mca.initializeMCA()
        self.parallelMatVec(correction)
        #print("RANK{}: benchmarking complete".format(self.mca.rank))

    def acquireMCAStats(self):
        self.mca.getMCAStats()

    def finalize(self):
        pass