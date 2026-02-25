# ==================================================================================================
# @author: Huynh Quang Nguyen Vo, Paritosh Ramanan
# @affiliation: Oklahoma State University
# @date: 2026-02-24
# ==================================================================================================

import math
import os, sys
import numpy as np

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
# CLASS DEFINITION
#===================================================================================================
class Root:
    def __init__(self, comm, x=None, mat=None):
        """
        Initialize a `Root` solver instance.

        Args:
            comm: MPI communicator object for distributed computing.
            x (optional): Input vector for the solver. Defaults to None.
            mat (optional): Input matrix for the solver. Defaults to None.

        Attributes:
            hardwareOn (bool): Hardware acceleration flag.
            scalingOn (bool): Scaling feature flag.
            deviceType (int): Device type from the environment variable `DT`.
            y_mem_result: Result from matrix-vector multiplication (MVM) by memristive devices.
            y_benchmark_result: Benchmark result from MVM by CPU.
            comm: MPI communicator for distributed operations.
            virtualizationOn (bool): Virtualization enabled flag (default: True).
            mca (RootMCA): MCA (memristor crossbar array) instance.
            origMatRows (int): Original matrix row count.
            origMatCols (int): Original matrix column count.
            cellRows (int): Cell decomposition row count.
            cellCols (int): Cell decomposition column count.
            mcaRows (int): MCA grid row count.
            mcaCols (int): MCA grid column count.
            origMat: Original matrix data.
            mcaGridRowCap (int): MCA grid row capacity.
            mcaGridColCap (int): MCA grid column capacity.
            maxVRows (int): Maximum virtual rows.
            maxVCols (int): Maximum virtual columns.
            x: Input vector.
            x_min: Minimum value in vector x.
            x_max: Maximum value in vector x.
            x_sum: Sum of vector x elements.
            virtualizer (dict): Dictionary for virtualization mappings.
        """
        self.hardwareOn = None
        self.scalingOn = None
        self.deviceType = int(os.environ["DT"])

        self.y_mem_result = None
        self.y_benchmark_result = None

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
        """
        Initialize the matrix and set up related configuration parameters.
        This method initializes the matrix through the `RootMCA` instance, which
        extracts matrix and system dimensions, calculates grid
        capacity, determines virtualization requirements, and optionally
        initializes the virtualizer if virtualization is enabled.

        Args:
            mat: The matrix to be initialized. This matrix is passed to the
                    matrix cell array for processing and storage.
        Returns:
            None
        Side Effects:
            - Initializes matrix cell array (mca) with the provided matrix
            - Sets original matrix dimensions (origMatRows, origMatCols)
            - Sets cell grid dimensions (cellRows, cellCols)
            - Sets matrix cell array grid dimensions (mcaRows, mcaCols)
            - Stores reference to original matrix (origMat)
            - Calculates grid capacity (mcaGridRowCap, mcaGridColCap)
            - Calculates maximum virtual matrix dimensions (maxVRows, maxVCols)
            - Initializes virtualizer dictionary
            - Calls initializeVirtualizer() if virtualizationOn is True
        """
        self.mca.initializeMatrix(mat)
        self.origMat = self.mca.mat

        # --- Extract matrix dimensions from the `RootMCA` instance after initialization ---
        self.origMatRows = self.mca.origMatRows
        self.origMatCols = self.mca.origMatCols

        # --- Extract cell grid dimensions from the `RootMCA` instance after initialization ---
        # These dimensions represent the number of cells per a single memristor crossbar array (MCA)
        self.cellRows = self.mca.cellRows
        self.cellCols = self.mca.cellCols

        # --- Extract MCA grid dimensions from the `RootMCA` instance after initialization ---
        # These dimensions represent how many MCAs present in the system 
        self.mcaRows = self.mca.mcaRows
        self.mcaCols = self.mca.mcaCols

        # --- Calculate the capacity of each MCA grid in terms of how many rows and columns of the original 
        # matrix it can accommodate ---
        self.mcaGridRowCap = self.mcaRows * self.cellRows
        self.mcaGridColCap = self.mcaCols * self.cellCols

        # --- Calculate the maximum virtual matrix dimensions based on the original matrix dimensions 
        # and the MCA grid capacity ---
        # This determines how many virtual tiles are needed to cover the entire original matrix when 
        # virtualization is enabled.
        self.maxVRows = math.ceil(float(self.origMatRows) / self.mcaGridRowCap)
        self.maxVCols = math.ceil(float(self.origMatCols) / self.mcaGridColCap)

        # --- Initialize the virtualizer dictionary, which will be used to store the mappings and data 
        # for virtualized matrix tiles when virtualization is enabled ---
        self.virtualizer = {}

        # --- If virtualization is enabled, call the method to initialize the virtualizer, which 
        # will decompose the original matrix into smaller tiles that fit within the MCA grid capacity 
        # for processing ---
        if self.virtualizationOn:
            self.initializeVirtualizer()

    def addCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum):
        """
        Reverse the effects of min-max scaling after a matrix-vector multiplication (MVM).
        
        This method corrects the scaled output vector by applying the inverse transformation
        using the scaling parameters from both the matrix and vector operands.
        
        Args:
            n (int): The number of elements or scaling factor.
            y (np.ndarray): The scaled output vector from matrix-vector multiplication.
            a_min (float): The minimum value used in scaling the matrix.
            a_max (float): The maximum value used in scaling the matrix.
            a_row_sum (np.ndarray): The sum of elements in each row of the matrix.
            x_min (float): The minimum value used in scaling the input vector.
            x_max (float): The maximum value used in scaling the input vector.
            x_sum (float): The sum of all elements in the input vector.
        
        Returns:
            np.ndarray: The corrected output vector with min-max scaling effects reversed.
        """
        Y = np.copy(y)
        for i in range(y.shape[0]):
            Y[i] = Y[i] * (a_max * x_max) + a_min * x_sum + x_min * a_row_sum[i] - n * a_min * x_min
        return Y

    def removeCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum):
        """
        Cancel the effects of the normalization reversal applied by `addCorrectionY`.
        
        This method re-applies the min-max scaling to the output vector, effectively reversing the 
        correction done by `addCorrectionY`. 
        It uses the same scaling parameters to transform the output vector back to the scaled domain.
        """
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

    def initializeX(self, x):
        """
        Prepares an input vector (`self.x`) for processing within a tiled MVM framework.

        Args:
            x (np.ndarray): The input vector to be initialized for memristive MVM. 
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

    def virtualParallelMatVec(self, i, j):
        """
        Performs a MVM for a specific tile of the input matrix and vector, and accumulates the 
        results into the appropriate segment of the output vector.

        Args:
            i (int): The index of the virtual row tile.
            j (int): The index of the virtual column tile.
        """
        data = np.array([i,j], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)

        self.mca.setMat(self.virtualizer[i,j]["mat"])
        self.mca.setX(self.virtualizer[i,j]["x"])
        y = self.mca.parallelMatVec()
        self.virtualizer[i]["y"] = self.virtualizer[i]["y"] + y

    def parallelMatVec(self, correction=False):
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
                    print(f"[INFO] ROOT: begin virtualParallelMatVec at MCA {i},{j}")
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
            mat_min, mat_max, mat_row_sum = __check_array_attributes__(self.mca.globalMat)
            print(f"[INFO] Matrix attributes for correction - min: {mat_min}, max: {mat_max}, row_sum (first 5): {mat_row_sum[:5]}")
            # Uncomment the following lines to force the matrix attributes
            # mat_min = float(self.mca.mat_min)
            # mat_max = float(self.mca.mat_max)
            # mat_row_sum = np.copy(self.mca.mat_row_sum)

            x_min, x_max, x_sum = self.x_min, self.x_max, self.x_sum
            print(f"[INFO] Vector attributes for correction - min: {x_min}, max: {x_max}, sum: {x_sum}")
            self.y_orig = self.addCorrectionY(self.origMatCols, self.y,
                          mat_min, mat_max, mat_row_sum,
                          x_min, x_max, x_sum)
            self.y_mem_result = np.copy(self.y_orig)
            print(f"[INFO] MELISO+ Result (with normalization reversal): \n {self.y_mem_result}")
            
            # Save the memristive MVM result (with normalization reversal)
            np.savetxt(__out_path__('y_mem_result_reversal_applied.txt'), self.y_mem_result, delimiter=',')
        else:
            self.y_mem_result = np.copy(self.y)
            print(f"[INFO] MELISO+ Result (without normalization reversal): \n {self.y_mem_result}")
            # Save the memristive MVM result (without normalization reversal)
            np.savetxt(__out_path__('y_mem_result_no_reversal.txt'), self.y_mem_result, delimiter=',')
        
        # Save the memristive MVM result
        np.savetxt(__out_path__('y_mem_result.txt'), self.y_mem_result, delimiter=',')



    def benchmarkMatVecParallel(self, hardwareOn=0, scalingOn=0, correction=False):
        """
        Compares the accuracy of an MVM performed by a memristive system with a ground truth (Numpy, etc.).

        Args:
            hardwareOn (int): Flag to indicate if hardware acceleration is used (default: 0).
            scalingOn (int): Flag to indicate if scaling is applied (default: 0).
            correction (bool): Flag to indicate if min-max scaling reversal should be applied before benchmarking (default: False).
        """
        assert self.mca.globalMat is not None, "Global matrix must be available for benchmarking."
        assert self.globalX is not None, "Global input vector must be available for benchmarking."

        # Placeholder for hardware and scaling flags, since both `Root` and `NonRoot` share the 
        # same `benchmarkMatVecParallel` method.
        self.hardwareOn = hardwareOn
        self.scalingOn = scalingOn

        # ------------------------------------------------------------------------------------------
        # Load the full matrix and vector and perform Numpy MVM for benchmarking
        # ------------------------------------------------------------------------------------------
        A_full = self.mca.globalMat   
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
            # Noted: For larger matrices, the CPU MVM may take a significant amount of time. Consider 
            # building the Numpy package with OpenBLAS or MKL for better performance.
            
            # TODO: Optimize Numpy MVM using multi-threading or GPU acceleration for larger matrices 
            # to reduce benchmarking time.
            y_cpu = A @ x
            print(f"[INFO] CPU MVM Result in the original domain: \n {y_cpu}")

            y_orig_mem = np.copy(y_mem_result)
            # Sanity check to see if `y_orig_mem` is not in the [0,1] range

            err_orig_domain = y_orig_mem - y_cpu
            den2 = max(np.linalg.norm(y_cpu, 2), 1e-15)
            deni = max(np.linalg.norm(y_cpu, np.inf), 1e-15)
            relL2_orig_domain   = np.linalg.norm(err_orig_domain, 2)    / den2
            relLinf_orig_domain = np.linalg.norm(err_orig_domain, np.inf) / deni
            
            # TODO: In some cases, the relative errors may be very large due to the nature of the 
            # original domain values, while in scaled domain, the errors are significantly smaller. 
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
            
            # In 99% of empirical cases, the relative errors in the scaled domain are significantly 
            # smaller. This is because the min-max scaling normalizes the values to a smaller range, 
            # which can help mitigate the effects of noise and non-idealities in the memristive MVM, 
            # leading to more accurate results compared to the original domain where values can be 
            # much larger and more susceptible to errors.
            print(f"[INFO] Comparing memristive MVM result to CPU MVM result in the scaled domain (without normalization reversal):")
            print(f"[INFO] Relative L2 error:  {relL2_scaled_domain}")
            print(f"[INFO] Relative Loo error:  {relLinf_scaled_domain}")
        
    def acquireMCAStats(self):
        """
        Retrieves and prints the MCA statistics (mean/stddev/maximum/minimum read and write 
        energy/latency)from the `RootMCA` instance.
        """
        self.mca.getMCAStats()

    def finalize(self):
        """
        Broadcast finalization signal to all processes.
        
        Creates and broadcasts a signal array to all processes in the communicator,
        indicating the finalization of operations. The signal is sent from the root
        process to ensure all processes receive the same termination indicator.
        
        The broadcast array [-1, -1] serves as a sentinel value to signal completion
        across the distributed computing environment.
        """
        data = np.array([-1, -1], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)

#---------------------------------------------------------------------------------------------------
# Internal methods for the Root class
#---------------------------------------------------------------------------------------------------
def __out_path__(name: str) -> str:
    """
    Define temporary output path for intermediate files.
    
    Args:
        name (str): The name of the file to be saved in the temporary directory.

    Returns:
        str: The full path to the output file, either in the temporary directory specified by the 
        `TMPDIR` environment variable or in the current directory if `TMPDIR` is not set.
    """
    base = os.environ.get("TMPDIR")
    if base:
        os.makedirs(base, exist_ok=True)
        return os.path.join(base, name)
    return name

def __check_array_attributes__(array):
    """
    Check and return key attributes of the input array for min-max scaling.
    
    Args:
        array (np.ndarray): The input array for which attributes are to be calculated.

    Returns:
        tuple: A tuple containing the minimum value, maximum value, and row-wise sum of the input 
        array, which are essential for performing min-max scaling and its reversal.
    """
    array_row_sum = np.sum(array, axis=1)
    array_min = array.min()
    array_max = array.ptp()
    return array_min, array_max, array_row_sum

def __minMax_Scale__(array):
    """
    Apply min-max scaling to the input array and return the scaled array along with the minimum and 
    maximum values.

    Args:
        array (np.ndarray): The input array to be scaled.
    
    Returns:
        tuple: A tuple containing the scaled array, the minimum value of the original array, and the 
        maximum value of the original array. The scaled array is normalized to the [0, 1] range.
    """
    array_min = array.min()
    array_max = array.ptp()
    return (array - array_min) / array_max, float(array_min), float(array_max)
    
