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
def _out_path(name: str) -> str:
    """Define temporary output path for intermediate files."""
    base = os.environ.get("TMPDIR")
    if base:
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, name)
    return name


def scaleMatrix(A):
    mat = A.astype(np.float64)
    mat_row_sum = np.sum(mat, axis=1)
    mat_min = mat.min()
    mat -= mat_min
    mat_max = mat.ptp()
    mat /= mat_max
    return mat,mat_min,mat_max,mat_row_sum

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

    # def removeCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum):
    #     Y = np.copy(y)
    #     for i in range(y.shape[0]):
    #         Y[i] = (Y[i] - a_min * x_sum - x_min * a_row_sum[i] + n * a_min * x_min) / (a_max * x_max)
    #     return Y

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

            self.globalX = np.copy(self.x)
            np.savetxt(_out_path('global_input_vec.txt'), self.x, delimiter=',')

            self.x_sum = np.sum(x)
            self.x, self.x_min, self.x_max, _ = self.mca.scaleMatrix(self.x)
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

    def parallelMatVec(self,correction=False):
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

        if correction:
            # self.y = self.addCorrectionY(self.origMatCols,
            #               self.y,
            #               self.mca.mat_min,
            #               self.mca.mat_max,
            #               self.mca.mat_row_sum,
            #               self.x_min,
            #               self.x_max,
            #               self.x_sum)
            pass

        self.y_mem_result = np.copy(self.y)
        print(f"[INFO] MELISO+ Result: \n {self.y_mem_result}")
        np.savetxt(_out_path('y_mem_result.txt'), self.y_mem_result, delimiter=',')

    # def benchmarkMatVec(self):
    #     print(f"[INFO] Benchmarking matrix size: {self.mca.mat.shape[0]} x {self.mca.mat.shape[1]}")
    #     print(f"[INFO] Benchmarking vector size: {self.mca.x.shape[0]} x {self.mca.x.shape[1]}")
    #     y = np.dot(self.mca.mat, self.mca.x)

    #     # Collect only the original column (in case of padding)
    #     self.y_benchmark_result = y[:self.mca.origMatRows]
    #     if self.y_benchmark_result.ndim > 1:
    #         self.y_benchmark_result = self.y_benchmark_result[:, 0]
    #         self.y_benchmark_unscaled_result = self.unscaleY(self.y_benchmark_result)
    #     print(f"[INFO] Numpy Unscaled Result size: \n {self.y_benchmark_unscaled_result.shape}")

    #     # Compute error:
    #     if self.y_mem_result is not None:
    #         if self.y_mem_result.ndim > 1: # Collect only the original column (in case of padding)
    #             self.y_mem_result = self.y_mem_result[:, 0]

    #         self.error = self.y_mem_result - self.y_benchmark_unscaled_result.flatten()
    #         rel_l2_norm = np.linalg.norm(self.error, ord=2) / \
    #             np.linalg.norm(self.y_benchmark_result, ord=2)
    #         print(f"[INFO] Relative L2-norm Error between MELISO+ and Numpy: {rel_l2_norm}")

    #         rel_loo_norm = np.linalg.norm(self.error, ord=np.inf) / \
    #             np.linalg.norm(self.y_benchmark_result, ord=np.inf)
    #         print(f"[INFO] Relative Loo-norm Error between MELISO+ and Numpy: {rel_loo_norm}")


    # def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0, correction= False):

    #     print(f"[INFO] Benchmarking matrix size: {self.mca.mat.shape[0]} x {self.mca.mat.shape[1]}")
    #     print(f"[INFO] Benchmarking vector size: {self.mca.x.shape[0]} x {self.mca.x.shape[1]}")
    #     y = self.mca.globalMat @ self.globalX
    #     print(f"[INFO] Numpy Unscaled Result: \n {y}")

    #     y_scaled = scaleMatrix(self.mca.globalMat)[0] @ scaleMatrix(self.globalX)[0]
    #     print(f"[INFO] Numpy Scaled Result: \n {y_scaled}")

    #     # Compute error:
    #     # if self.y_mem_result is None:
    #     np.loadtxt_exists = os.path.isfile('y_mem_result.txt')
    #     if np.loadtxt_exists:
    #         print(f"[INFO] Loading existing MELISO+ Result for benchmarking.")
    #         self.y_mem_result = np.loadtxt('y_mem_result.txt')
    #         print(f"[INFO] MELISO+ Loaded Result size: \n {self.y_mem_result.shape}")
    #         print(f"[INFO] MELISO+ Loaded Result: \n {self.y_mem_result}")
    #         self.y_mem_result_unscaled = self.unscaleY(self.y_mem_result)


    #         # Unscaled result error computation
    #         self.error = self.y_mem_result_unscaled - y
    #         print(f"[INFO] Unscaled Error vector: \n {self.error}")

    #         # Compute relative error norms
    #         rel_l2_norm = np.linalg.norm(self.error, ord=2) / \
    #             np.linalg.norm(y, ord=2)
    #         print(f"[INFO] Unscaled Relative L2-norm Error between MELISO+ and Numpy: {rel_l2_norm}")

    #         rel_loo_norm = np.linalg.norm(self.error, ord=np.inf) / \
    #             np.linalg.norm(y, ord=np.inf)
    #         print(f"[INFO] Unscaled Relative Loo-norm Error between MELISO+ and Numpy: {rel_loo_norm}")

    #         # Scaled result error computation
    #         self.error_scaled = self.y_mem_result - y_scaled
    #         print(f"[INFO] Scaled Error vector: \n {self.error_scaled}")
    #         rel_l2_norm = np.linalg.norm(self.error_scaled, ord=2) / \
    #             np.linalg.norm(y_scaled, ord=2)
    #         print(f"[INFO] Scaled Relative L2-norm Error between MELISO+ and Numpy: {rel_l2_norm}")
    #         rel_loo_norm = np.linalg.norm(self.error_scaled, ord=np.inf) / \
    #             np.linalg.norm(y_scaled, ord=np.inf)
    #         print(f"[INFO] Scaled Relative Loo-norm Error between MELISO+ and Numpy: {rel_loo_norm}")

    def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0, correction=False):
        A_full = self.mca.globalMat
        x_full = self.globalX
        if x_full.ndim > 1:
            x_full = x_full.reshape(-1)         # (n_full,)

        n_cols = int(getattr(self, "origMatCols", A_full.shape[1]))
        A = A_full[:, :n_cols]                  # (m, n)
        x = x_full[:n_cols]                     # (n,)

        m, n = A.shape
        print(f"[INFO] Benchmarking matrix (used): {m} x {n}")
        print(f"[INFO] Benchmarking vector (used): {x.shape[0]}")

        # ----------------------------
        # 1) CPU references
        # ----------------------------
        # Unscaled/original CPU result
        y_cpu = A @ x                           # (m,)
        print(f"[INFO] Numpy Unscaled Result (A@x) ready")

        # Helper: global min–max scale to [0,1]
        def minmax_scale(arr):
            amin = arr.min()
            arng = arr.max() - amin
            if arng == 0:
                return np.zeros_like(arr), float(amin), 0.0
            return (arr - amin) / arng, float(amin), float(arng)

        # Scaled-domain CPU result (what the device computes)
        # IMPORTANT: x must be column-shaped for matrix multiply
        A_scaled, a_min, a_rng = minmax_scale(A)
        x_col = x.reshape(-1, 1)
        x_scaled, x_min, x_rng = minmax_scale(x_col)
        x_scaled = x_scaled.reshape(-1)         # back to (n,)
        y_scaled_cpu = A_scaled @ x_scaled      # (m,)
        print(f"[INFO] Numpy Scaled Result (A_scaled@x_scaled) ready")

        # ----------------------------
        # 2) Load memristor output (assumed scaled-domain)
        # ----------------------------
        res_path = _out_path("y_mem_result.txt")
        if not os.path.isfile(res_path):
            print("[ERROR] y_mem_result.txt not found — run the memristor multiply first.")
            return

        y_mem_result = np.loadtxt(res_path)  # robust for whitespace/newlines
        if y_mem_result.ndim > 1:
            y_mem_result = y_mem_result[:, 0]
        if y_mem_result.shape[0] != m:
            print(f"[ERROR] y_mem_result length {y_mem_result.shape[0]} != rows of A {m}")
            return
        print(f"[INFO] Loaded mem y_scaled with shape: {y_mem_result.shape}")

        # ----------------------------
        # 3) Scaled-domain error
        # ----------------------------
        err_scaled = y_mem_result - y_scaled_cpu
        den2 = max(np.linalg.norm(y_scaled_cpu, 2), 1e-15)
        deni = max(np.linalg.norm(y_scaled_cpu, np.inf), 1e-15)
        relL2_scaled  = np.linalg.norm(err_scaled, 2)    / den2
        relLinf_scaled= np.linalg.norm(err_scaled, np.inf) / deni
        print(f"[INFO] Scaled Relative L2:  {relL2_scaled}")
        print(f"[INFO] Scaled Relative Loo:  {relLinf_scaled}")

        # ----------------------------
        # 4) Exact inverse of min–max scaling to original domain
        #     y = (a_rng*x_rng)*y_scaled + x_min*(A@1) + a_min*(sum(x))*1 - (a_min*x_min*n)*1
        # ----------------------------
        A_row_sum = A.sum(axis=1)               # (m,)
        sum_x = float(x.sum())
        ones_m = np.ones(m)

        if a_rng == 0 and x_rng == 0:
            y_mem_unscaled = (a_min * sum_x) * ones_m
        elif a_rng == 0:
            y_mem_unscaled = (a_min * sum_x) * ones_m
        elif x_rng == 0:
            y_mem_unscaled = x_min * A_row_sum
        else:
            y_mem_unscaled = (a_rng * x_rng) * y_mem_result \
                            + x_min * A_row_sum \
                            + a_min * sum_x * ones_m \
                            - (a_min * x_min * n) * ones_m

        # ----------------------------
        # 5) Original-domain error
        # ----------------------------
        err_unscaled = y_mem_unscaled - y_cpu
        den2_u = max(np.linalg.norm(y_cpu, 2), 1e-15)
        deni_u = max(np.linalg.norm(y_cpu, np.inf), 1e-15)
        relL2_unscaled  = np.linalg.norm(err_unscaled, 2)    / den2_u
        relLinf_unscaled= np.linalg.norm(err_unscaled, np.inf) / deni_u
        print(f"[INFO] Unscaled Relative L2: {relL2_unscaled}")
        print(f"[INFO] Unscaled Relative Loo: {relLinf_unscaled}")

        # Helpful context: scaling amplification factor
        print(f"[INFO] a_min={a_min}, a_max={a_min + a_rng}, x_min={x_min}, x_max={x_min + x_rng}, "
            f"(a_rng*x_rng)={a_rng * x_rng}")
        
    def acquireMCAStats(self):
        self.mca.getMCAStats()

    def finalize(self):
        data = np.array([-1, -1], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)

    def unscaleY(self, y_scaled):
        a_max, a_min = self.mca.globalMat.max(), self.mca.globalMat.min()
        x_max, x_min = self.globalX.max(), self.globalX.min()

        a_range = a_max - a_min
        x_range = x_max - x_min
        n = self.mca.globalMat.shape[1]
        ones = np.ones(self.mca.globalMat.shape[0])

        term1 = a_range * x_range * y_scaled
        term2 = x_min * (self.mca.globalMat @ np.ones(n))
        term3 = a_min * (np.sum(self.globalX) * ones)
        term4 = -a_min * x_min * n * ones
        y = term1 + term2 + term3 + term4
        return y


    
