import numpy as np
import math
import os,sys

if "MELISO_SRC_PATH" in os.environ.keys():
    if not os.isdir(os.environ["MELISO_SRC_PATH"]):
        raise Exception("Env Var {} MELISO_SRC_PATH is invalid!".format(os.environ["MELISO_SRC_PATH"]))
    sys.path.append(os.environ["MELISO_SRC_PATH"])
else:
    sys.path.append("../../")
from src.core.RootMCA import RootMCA

class Root:
    def __init__(self,comm,x=None,mat=None):

        self.hardwareOn = None
        self.scalingOn = None
        
        self.y_mem_result = None
        self.y_benchmark_result = None
        self.error = None
        self.comm = comm
        self.virtualizationOn = True
        self.mca = RootMCA(self.comm) #,mat=mat,set_mat=False)

        self.origMatRows = None
        self.origMatCols = None

        self.cellRows = None
        self.cellCols = None

        self.mcaRows = None
        self.mcaCols = None

        self.origMat = None

        self.mcaGridRowCap = None
        self.mcaGridColCap = None

        self.maxVRows = None #math.ceil(float(self.origMatRows) / self.mcaGridRowCap)
        self.maxVCols = None #math.ceil(float(self.origMatCols) / self.mcaGridColCap)


        self.x = None  #
        self.x_min = None
        self.x_max = None
        self.x_sum = None

        self.virtualizer = {}

        self.maxVRows = None
        self.maxVCols = None

        self.initializeMat(mat)

        # can implement a variety of x vector initializations here.
        self.initializeX(x)

    def initializeMat(self,mat):
        self.mca.initializeMatrix(mat)

        self.origMatRows = self.mca.origMatRows
        self.origMatCols = self.mca.origMatCols

        self.cellRows = self.mca.cellRows
        self.cellCols = self.mca.cellCols

        self.mcaRows = self.mca.mcaRows
        self.mcaCols = self.mca.mcaCols

        self.origMat = self.mca.mat

        self.mcaGridRowCap = self.mcaRows * self.cellRows
        self.mcaGridColCap = self.mcaCols * self.cellCols

        self.maxVRows = math.ceil(float(self.origMatRows) / self.mcaGridRowCap)
        self.maxVCols = math.ceil(float(self.origMatCols) / self.mcaGridColCap)

        self.virtualizer = {}

        if self.virtualizationOn:
            self.initializeVirtualizer()

    def addCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum):
        Y = np.copy(y)
        for i in range(y.shape[0]):
            Y[i] = Y[i] * (a_max * x_max) + a_min * x_sum + x_min * a_row_sum[i] - n * a_min * x_min
        return Y

    def removeCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum):
        Y = np.copy(y)
        for i in range(y.shape[0]):
            Y[i] = (Y[i] - a_min * x_sum - x_min * a_row_sum[i] + n * a_min * x_min) / (a_max * x_max)
        return Y

    def initializeVirtualizer(self):
        # do all matrix chunking and preprocessing here
        for i in range(self.maxVRows):
            self.virtualizer[i] = {}
            start_vRow = self.mcaGridRowCap*i
            vRows = self.mcaGridRowCap
            if self.origMatRows - self.mcaGridRowCap*i < vRows:
                vRows = self.origMatRows - self.mcaGridRowCap*(i)

            end_vRow = start_vRow + vRows

            for j in range(self.maxVCols):
                start_vCol = self.mcaGridColCap*j
                vCols = self.mcaGridColCap
                if self.origMatCols - self.mcaGridColCap * j < vCols:
                    vCols = self.origMatCols - self.mcaGridColCap *j

                end_vCol = start_vCol + vCols

                self.virtualizer[i,j] = {}
                self.virtualizer[i,j]["rc_limits"] = [[start_vRow,end_vRow],[start_vCol,end_vCol]]
                self.virtualizer[i,j]["mat"] = self.origMat[start_vRow:end_vRow,start_vCol:end_vCol]

                if self.x is not None:
                    self.virtualizer[i,j]["x"] =  np.copy(self.x.reshape(self.x.shape[0],1)[start_vCol:end_vCol])

            self.virtualizer[i]["y"] = np.zeros(end_vRow-start_vRow,dtype=np.float64)

    def initializeX(self,x):
        if x is not None:
            self.x = x.reshape(x.shape[0], 1)[:self.origMatCols]
            self.x_sum = np.sum(x)
            self.x,self.x_min,self.x_max,_ = self.mca.scaleMatrix(self.x)
            for i in range(self.maxVRows):
                for j in range(self.maxVCols):
                    sc = self.virtualizer[i, j]["rc_limits"][1][0]
                    ec = self.virtualizer[i, j]["rc_limits"][1][1]
                    self.virtualizer[i, j]["x"] = np.copy(self.x.reshape(self.x.shape[0], 1)[sc:ec,:])

    def virtualParallelMatVec(self,i,j):
        #set the matrix
        data = np.array([i,j], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)
        #print("ROOT: broadcasted {} {}".format(i, j))

        self.mca.setMat(self.virtualizer[i,j]["mat"])

        self.mca.setX(self.virtualizer[i,j]["x"])

        y = self.mca.parallelMatVec()

        #print("ROOT: after parallelMatvec {} {}".format(i, j))

        self.virtualizer[i]["y"] = self.virtualizer[i]["y"] + y

    def parallelMatVec(self,type="mca",correction=False):
        if self.virtualizationOn:
            print(self.maxVRows,self.maxVCols)
            y = np.zeros(self.origMatRows, dtype=np.float64)
            for i in range(self.maxVRows):
                for j in range(self.maxVCols):
                    #print("ROOT: begin virtualParallelMatVec {},{}".format(i,j))
                    self.virtualParallelMatVec(i,j)

                sr = self.virtualizer[i,0]["rc_limits"][0][0]
                er = self.virtualizer[i,0]["rc_limits"][0][1]
                y[sr:er] = self.virtualizer[i]["y"]
                self.virtualizer[i]["y"] = np.zeros(er - sr, dtype=np.float64)
            self.y = y
        else:
            self.mca.setX(self.x)
            self.y = self.mca.parallelMatVec()
        #print("Correction:{}".format(correction))
        if correction:
            self.y = self.addCorrectionY(self.origMatCols,
                          self.y,
                          self.mca.mat_min,
                          self.mca.mat_max,
                          self.mca.mat_row_sum,
                          self.x_min,
                          self.x_max,
                          self.x_sum)

        if type == "benchmark":
            self.y_benchmark_result = np.copy(self.y)
            print(f"\nBenchmarked Result: \n {self.y_benchmark_result}")
        else:
            if "DT" in os.environ.keys():
                self.device_type = int(os.environ["DT"])
                
            self.y_mem_result = (self.device_type + 1) * np.copy(self.y)
            print(f"\nMultiplication Result: \n {self.y_mem_result}")

    def benchmarkMatVec(self):
        y = np.dot(self.mca.mat, self.mca.x)
        self.y_benchmark_result = y[:self.mca.origMatRows]
        self.error = self.y_mem_result - self.y_benchmark_result.flatten()
        if self.y_mem_result is not None:
            print("\nElement-wise Error: \n", self.error)
            print(f"L2-norm Error: {np.linalg.norm(self.error, ord=2)}")
            print(f"Loo-norm Error: {np.linalg.norm(self.error, ord=np.inf)}")

    def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0, correction= False):

        self.parallelMatVec(type="benchmark", correction=correction)
        if self.y_mem_result is not None:
            self.error = self.y_mem_result - self.y_benchmark_result

            print("\nElement-wise Error: \n", self.error)
            print(f"L2-norm Error: {np.linalg.norm(self.error, ord=2)}")
            print(f"Loo-norm Error: {np.linalg.norm(self.error, ord=np.inf)}")

    def acquireMCAStats(self):
        self.mca.getMCAStats()
        print("AllMCAStats\n")
        print(self.mca.allMCAStats)

    def finalize(self):
        data = np.array([-1, -1], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)
