import numpy as np
import math
import os,sys

#===================================================================================================
# Import MELISO+ core
#===================================================================================================
if "MELISO_SRC_PATH" in os.environ.keys():
    if not os.path.isdir(os.environ["MELISO_SRC_PATH"]):
        raise Exception(f"Env Var {os.environ['MELISO_SRC_PATH']} is invalid!")
    sys.path.append(os.environ["MELISO_SRC_PATH"])
else:
    sys.path.append("../../")
from src.core.RootMCA import RootMCA


#===================================================================================================
# Utility functions
#===================================================================================================
def __out_path__(name: str) -> str:
    """Define temporary output path for intermediate files."""
    base = os.environ.get("TMPDIR")
    if base:
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, name)
    return name

def __check_array_attributes__(array):
    array_row_sum = np.sum(array, axis=1)
    array_min = array.min()
    array_max = array.ptp()
    return array_min, array_max, array_row_sum

def __minMax_Scale__(array):
    array_min = array.min()
    array_range = array.max() - array_min
    if array_range == 0:
        return np.zeros_like(array), float(array_min), 0.0
    return (array - array_min) / array_range, float(array_min), float(array_range)

#===================================================================================================
# CLASS DEFINITION
#===================================================================================================
class Root:
    def __init__(self,comm,x=None,mat=None):

        self.hardwareOn = None
        self.scalingOn = None
        self.deviceType = int(os.environ["DT"])
        self.y_mem_result = None
        self.y_benchmark_result = None
        self.error = None
        self.comm = comm
        self.virtualizationOn = True
        self.mca = RootMCA(self.comm) 

        self.origMatRows = None
        self.origMatCols = None

        self.cellRows = None
        self.cellCols = None

        self.mcaRows = None
        self.mcaCols = None

        self.origMat = None

        self.mcaGridRowCap = None
        self.mcaGridColCap = None

        self.maxVRows = None 
        self.maxVCols = None 


        self.x = x
        self.x_min = None
        self.x_max = None
        self.x_sum = None

        self.virtualizer = {}

        self.maxVRows = None
        self.maxVCols = None

        self.initializeMat(mat)
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
        """Reverse the effects of the min-max scaling after a matrix-vector multiplication (MVM) has been performed."""
        Y = np.copy(y)
        for i in range(y.shape[0]):
            Y[i] = Y[i] * (a_max * x_max) + a_min * x_sum + x_min * a_row_sum[i] - n * a_min * x_min
        return Y

    def removeCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum):
        """Cancel the effects of the normalization reversal applied by `addCorrectionY`."""
        Y = np.copy(y)
        for i in range(y.shape[0]):
            Y[i] = (Y[i] - a_min * x_sum - x_min * a_row_sum[i] + n * a_min * x_min) / (a_max * x_max)
        return Y

    def initializeVirtualizer(self):
        """
        Decompose the input matrix (`self.origMat`) and an input vector (`self.x`) into smaller 
        titles (or "tiles") through the tiling process.
        """
        assert self.origMat is not None, "Matrix must be initialized before virtualizer."
        assert self.x is not None, "Input vector x must be initialized before virtualizer."
        assert self.maxVRows is not None and self.maxVCols is not None, "Virtualizer dimensions must be set."
        assert self.mcaGridRowCap is not None and self.mcaGridColCap is not None, "MCA grid capacities must be set."
        assert self.origMatRows is not None and self.origMatCols is not None, "Original matrix dimensions must be set."
        
        for i in range(self.maxVRows):
            # Row chunking
            self.virtualizer[i] = {}
            start_vRow = self.mcaGridRowCap * i
            vRows = self.mcaGridRowCap
            if self.origMatRows - self.mcaGridRowCap * i < vRows:
                vRows = self.origMatRows - self.mcaGridRowCap*(i)

            end_vRow = start_vRow + vRows

            for j in range(self.maxVCols):
                # Column chunking
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
            
            # Initialize output buffer 
            self.virtualizer[i]["y"] = np.zeros(end_vRow-start_vRow,dtype=np.float64)

    def initializeX(self,x):
        """
        Prepares an input vector (`self.x`) for processing within a tiled MVM framework. 
        """
        assert self.origMatCols is not None, "Original matrix dimensions must be set before initializing the input vector."
        assert self.maxVRows is not None and self.maxVCols is not None, "Virtualizer dimensions must be set before initializing x."

        if x is not None:
            self.x = x.reshape(x.shape[0], 1)[:self.origMatCols]

            self.globalX = np.copy(self.x)
            np.savetxt(__out_path__('global_input_vec.txt'), self.x, delimiter=',')

            self.x_sum = float(np.sum(self.globalX))
            self.x, self.x_min, self.x_max, _ = self.mca.scaleMatrix(self.x)
            for i in range(self.maxVRows):
                for j in range(self.maxVCols):
                    sc = self.virtualizer[i, j]["rc_limits"][1][0]
                    ec = self.virtualizer[i, j]["rc_limits"][1][1]
                    self.virtualizer[i, j]["x"] = np.copy(self.x.reshape(self.x.shape[0], 1)[sc:ec,:])

    def virtualParallelMatVec(self,i,j):
        """
        Performs a MVM for a specific tile of the input matrix and vector, and accumulates the results into the appropriate segment of the output vector.
        """
        data = np.array([i,j], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)

        self.mca.setMat(self.virtualizer[i,j]["mat"])
        self.mca.setX(self.virtualizer[i,j]["x"])
        y = self.mca.parallelMatVec()
        self.virtualizer[i]["y"] = self.virtualizer[i]["y"] + y

    def parallelMatVec(self,correction=False):
        """
        Executes a MVM across the entire input matrix and vector, utilizing tiling and virtualization if enabled.
        """
        assert self.origMatRows is not None and self.origMatCols is not None, "Original matrix dimensions must be set before performing MVM."
        assert self.maxVRows is not None and self.maxVCols is not None, "Virtualizer dimensions must be set before performing MVM."

        if self.virtualizationOn:
            print(f"Virtualization Enabled: Max Rows {self.maxVRows}, Max Cols {self.maxVCols}")

            y = np.zeros(self.origMatRows, dtype=np.float64)
            for i in range(self.maxVRows):
                for j in range(self.maxVCols):
                    print(f"[INFO]ROOT: begin virtualParallelMatVec at MCA {i},{j}")
                    self.virtualParallelMatVec(i,j)

                sr = self.virtualizer[i,0]["rc_limits"][0][0]
                er = self.virtualizer[i,0]["rc_limits"][0][1]
                y[sr:er] = self.virtualizer[i]["y"]
                self.virtualizer[i]["y"] = np.zeros(er - sr, dtype=np.float64)
            self.y = y
        else:
            self.mca.setX(self.x)
            self.y = self.mca.parallelMatVec()

        if correction == True:
            # Sanity check for correction parameters
            mat_min, mat_max, mat_row_sum = __check_array_attributes__(self.origMat)
            x_min, x_max, x_sum = self.x_min, self.x_max, self.x_sum
            self.y_orig = self.addCorrectionY(self.origMatCols, self.y,
                          mat_min, mat_max, mat_row_sum,
                          x_min, x_max, x_sum)
            self.y_mem_result = np.copy(self.y_orig)
            print(f"[INFO] MELISO+ Result (with normalization reversal): \n {self.y_mem_result}")
        else:
            self.y_mem_result = np.copy(self.y)
            print(f"[INFO] MELISO+ Result (without normalization reversal): \n {self.y_mem_result}")

        # Save the memristive MVM result
        np.savetxt(__out_path__('y_mem_result.txt'), self.y_mem_result, delimiter=',')

    def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0, correction=False):
        """
        Compares the accuracy of an MVM performed by a memristive system with a ground truth (Numpy, etc.).
        """
        assert self.origMat is not None, "Global matrix must be available for benchmarking."
        assert self.globalX is not None, "Global input vector must be available for benchmarking."

        # ------------------------------------------------------------------------------------------
        # Load the full matrix and vector and perform Numpy MVM for benchmarking
        # ------------------------------------------------------------------------------------------
        A_full = self.origMat   
        n_cols = int(getattr(self, "origMatCols", A_full.shape[1]))
        A = A_full[:, :n_cols]                  

        x_full = self.globalX
        if x_full.ndim > 1:
            x_full = x_full.reshape(-1)         # (n_full,)
        x = x_full[:n_cols]                     # (n,)

        # ------------------------------------------------------------------------------------------
        # Load memristive MVM output and apply normalization reversal if correction is enabled
        # ------------------------------------------------------------------------------------------
        result_path = __out_path__("y_mem_result.txt")
        if not os.path.isfile(result_path):
            print("[ERROR] y_mem_result.txt not found - run the memristive MVM first.")
            return

        y_mem_result = np.loadtxt(result_path)  
        if y_mem_result.ndim > 1:
            y_mem_result = y_mem_result[:, 0]
        if y_mem_result.shape[0] != self.origMatRows:
            print(f"[ERROR] y_mem_result length {y_mem_result.shape[0]} != rows of A {self.origMatRows}")
            return
        print(f"[INFO] Loaded y_mem_result with shape: {y_mem_result.shape}")
        
        # ------------------------------------------------------------------------------------------
        # Compute and print relative errors based on whether normalization reversal was applied
        # ------------------------------------------------------------------------------------------
        if correction == True:
            y_cpu = A @ x
            print(f"[INFO] CPU MVM Result in the original domain: \n {y_cpu}")

            y_orig_mem = np.copy(y_mem_result)
            # Sanity check to see if `y_orig_mem` is not in the [0,1] range

            err_orig_domain = y_orig_mem - y_cpu
            den2 = max(np.linalg.norm(y_cpu, 2), 1e-15)
            deni = max(np.linalg.norm(y_cpu, np.inf), 1e-15)
            relL2_orig_domain   = np.linalg.norm(err_orig_domain, 2)    / den2
            relLinf_orig_domain = np.linalg.norm(err_orig_domain, np.inf) / deni
            
            print(f"[INFO] Comparing memristive MVM result to CPU MVM result in the original domain (after normalization reversal):")
            print(f"[INFO] Relative L2 error:  {relL2_orig_domain}")
            print(f"[INFO] Relative Loo error:  {relLinf_orig_domain}")
        else:
            A_scaled_cpu, _ , _ = __minMax_Scale__(A)
            x_scaled_cpu, _ , _ = __minMax_Scale__(x)
            y_scaled_cpu = A_scaled_cpu @ x_scaled_cpu

            print(f"[INFO] CPU MVM Result after min-max scaling: \n {y_scaled_cpu}")
            err_scaled_domain = y_mem_result - y_scaled_cpu
            den2 = max(np.linalg.norm(y_scaled_cpu, 2), 1e-15)
            deni = max(np.linalg.norm(y_scaled_cpu, np.inf), 1e-15)
            relL2_scaled_domain   = np.linalg.norm(err_scaled_domain, 2)    / den2
            relLinf_scaled_domain = np.linalg.norm(err_scaled_domain, np.inf) / deni
            
            print(f"[INFO] Comparing memristive MVM result to CPU MVM result in the scaled domain (without normalization reversal):")
            print(f"[INFO] Relative L2 error:  {relL2_scaled_domain}")
            print(f"[INFO] Relative Loo error:  {relLinf_scaled_domain}")
        
    def acquireMCAStats(self):
        self.mca.getMCAStats()

    def finalize(self):
        data = np.array([-1, -1], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)

    
