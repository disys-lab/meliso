import os
import sys
import numpy as np

# Adjust the Python path to include the MELISO source path
if "MELISO_SRC_PATH" in os.environ.keys():
    meliso_src_path = os.environ["MELISO_SRC_PATH"]

    if not os.path.isdir(meliso_src_path):
        raise Exception(f"MELISO_SRC_PATH: '{meliso_src_path}' is invalid!")
    sys.path.append(meliso_src_path)
else:
    sys.path.append("../../")

from src.core.RootMCA import RootMCA

# Set the report path from the environment variable or default to 'report.txt'
REPORT_PATH = os.environ.get("REPORT_PATH", "report.txt")

class Root:
    def __init__(self, comm, x=None, mat=None) -> None:
        """
        The Root class handles parallel matrix-vector multiplication (MVM) with virtualization support.
        """

        # Setup MCA operation
        self.comm = comm
        self.virtualizationOn = True
        self.mca = RootMCA(self.comm)
        self.hardwareOn = 0
        self.scalingOn = 0
        self.deviceType = int(os.environ.get("DT", 0))

        # Placeholder for storing virtualized chunks of matrix and vectors
        self.virtualizer = {}

        # Placeholders for multiplication results
        self.y_mem_result = None
        self.y_benchmark_result = None
        self.error = None

        # Placeholders for original input matrix dimensions
        self.origMat = None
        self.origMatRows = None
        self.origMatCols = None

        # Placeholders for original input vector dimensions
        self.x = None
        self.x_min = None
        self.x_max = None
        self.x_sum = None

        # Placeholders for cell dimensions
        self.cellRows = None
        self.cellCols = None

        # Placeholders for MCA grid dimensions
        self.mcaRows = None
        self.mcaCols = None

        # Placeholders for MCA grid capacities
        self.mcaGridRowCap = None
        self.mcaGridColCap = None

        # Placeholders for maximum virtual rows and columns
        self.maxVRows = None
        self.maxVCols = None

        # Initialize input matrix
        self.initializeMat(mat)

        # Initialize the virtualizer
        if self.virtualizationOn:
            self.initializeVirtualizer()

        # Initialize input vector
        self.initializeX(x)

        return None

    def initializeMat(self, mat)-> None:
        """
        Initializes the input matrix, denoted by A, and sets up virtualization if enabled.
        """
        
        # Initialize the matrix
        self.mca.initializeMatrix(mat)
        self.origMat = self.mca.mat

        # Set original matrix dimensions
        self.origMatRows = self.mca.origMatRows
        self.origMatCols = self.mca.origMatCols
        
        # Set cell dimensions
        self.cellRows = self.mca.cellRows
        self.cellCols = self.mca.cellCols
        
        # Set MCA grid dimensions
        self.mcaRows = self.mca.mcaRows
        self.mcaCols = self.mca.mcaCols

        # Compute MCA grid capacities
        self.mcaGridRowCap = self.mcaRows * self.cellRows
        self.mcaGridColCap = self.mcaCols * self.cellCols

        # Compute maximum virtual rows and columns
        self.maxVRows = np.ceil(float(self.origMatRows) / self.mcaGridRowCap)
        self.maxVCols = np.ceil(float(self.origMatCols) / self.mcaGridColCap)

        self.virtualizer = {}
        if self.virtualizationOn:
            self.initializeVirtualizer()

        return None

    def addCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum) -> np.ndarray:
        """
        Add correction to the computed vector, denoted by y = Ax.
        """
        Y = y * (a_max * x_max) + a_min * x_sum + x_min * a_row_sum - n * a_min * x_min
        return Y

    def removeCorrectionY(self, n, y, a_min, a_max, a_row_sum, x_min, x_max, x_sum) -> np.ndarray:
        """
        Remove correction to the computed y = Ax vector.
        """
        Y = (y - a_min * x_sum - x_min * a_row_sum + n * a_min * x_min) / (a_max * x_max)
        return Y
    
    def initializeVirtualizer(self)-> None:
        """
        Initialize the virtualizer by dividing the input matrix A into chunks and preprocessing.
        """

        # Loop over virtual rows
        for i in range(int(self.maxVRows)):
            self.virtualizer[i] = {}
            start_vRow = self.mcaGridRowCap * i
            vRows = min(self.mcaGridRowCap, self.origMatRows - start_vRow)
            end_vRow = start_vRow + vRows

            # Loop over virtual columns
            for j in range(int(self.maxVCols)):
                start_vCol = self.mcaGridColCap * j
                vCols = min(self.mcaGridColCap, self.origMatCols - start_vCol)
                end_vCol = start_vCol + vCols

                self.virtualizer[i, j] = {}
                self.virtualizer[i, j]["rc_limits"] = [[start_vRow, end_vRow], [start_vCol, end_vCol]]
                self.virtualizer[i, j]["mat"] = self.origMat[start_vRow:end_vRow, start_vCol:end_vCol]

                if self.x is not None:
                    self.virtualizer[i, j]["x"] = self.x[start_vCol:end_vCol]

            self.virtualizer[i]["y"] = np.zeros(vRows, dtype=np.float64)

        return None
    
    def initializeX(self, x)-> None:
        """
        Initialize the input vector, denoted by x, and scales it if necessary.
        """
        if x is not None:
            
            # Reshape x to be a column vector and truncate to match matrix dimensions
            self.x = x.reshape(-1, 1)[:self.origMatCols]
            self.x_sum = np.sum(x)
            
            # Scale x using the RootMCA's scaling method
            self.x, self.x_min, self.x_max, _ = self.mca.scaleMatrix(self.x)

            # Update x in the virtualizer
            for i in range(int(self.maxVRows)):
                for j in range(int(self.maxVCols)):
                    sc = self.virtualizer[i, j]["rc_limits"][1][0]
                    ec = self.virtualizer[i, j]["rc_limits"][1][1]
                    self.virtualizer[i, j]["x"] = self.x[sc:ec]
        return None

    def virtualParallelMatVec(self, i, j)-> None:
        """
        Perform parallel MVM on a virtual chunk.
        """

        # Broadcast the indices to all processes
        data = np.array([i, j], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)

        # Set the matrix and vector chunk in RootMCA
        self.mca.setMat(self.virtualizer[i,j]["mat"])
        self.mca.setX(self.virtualizer[i,j]["x"])

        # Perform parallel MVM
        y_chunk = self.mca.parallelMatVec()
        self.virtualizer[i]["y"] += y_chunk

        return None

    def parallelMatVec(self, type="mca", correction=False)-> None:
        """
        Perform parallel MVM, with optional correction.
        """

        if self.virtualizationOn:
            y = np.zeros(self.origMatRows, dtype=np.float64)

            # Iterate over virtual rows and columns
            for i in range(int(self.maxVRows)):
                for j in range(int(self.maxVCols)):
                    self.virtualParallelMatVec(i, j)

                # Extract the computed vector y for the virtual row
                sr = self.virtualizer[i, 0]["rc_limits"][0][0]
                er = self.virtualizer[i, 0]["rc_limits"][0][1]
                y[sr:er] = self.virtualizer[i]["y"]

                # Reset the y accumulator for the next iteration
                self.virtualizer[i]["y"] = np.zeros(er - sr, dtype=np.float64)
            self.y = y
        else:
            self.mca.setX(self.x)
            self.y = self.mca.parallelMatVec()

        # Apply correction if needed
        if correction:
            self.y = self.addCorrectionY(self.origMatCols, self.y,
                          self.mca.mat_min, self.mca.mat_max, self.mca.mat_row_sum, 
                          self.x_min, self.x_max, self.x_sum)

        if type == "benchmark":
            self.y_benchmark_result = np.copy(self.y)
            print(f"\nBenchmarked Result: \n{self.y_benchmark_result}")
        else:
            # Adjust the result based on device type
            self.y_mem_result = (self.deviceType + 1) * np.copy(self.y)
            print(f"\nMultiplication Result: \n{self.y_mem_result}")

        return None

    def benchmarkMatVec(self)-> None:
        """
        Perform MVM using NumPy for benchmarking and compute the error with respect to y_mem_result.
        """

        # Compute benchmark MVM and truncate to match matrix dimensions 
        y = np.dot(self.mca.mat, self.mca.x)
        self.y_benchmark_result = y[:self.mca.origMatRows]

        if self.y_mem_result is not None:
            self.error = self.y_mem_result - self.y_benchmark_result
            print("\nElement-wise Error: \n", self.error)

            # Compute relative L-infinity norm error
            relative_error = (np.linalg.norm(self.error, ord=np.inf) /
                              np.linalg.norm(self.y_mem_result, ord=np.inf))
            print(f"Relative Loo-norm Error: {relative_error}\n")

            # Store the computed relative L-infinity norm error
            with open(REPORT_PATH, "a+") as file:
                file.write(f"Relative Loo-norm Error: {relative_error}\n")
        return None

    def benchmarkMatVecParallel(self, hardwareOn, scalingOn, correction= False)-> None:
        """
        Perform parallel MVM using NumPy for benchmarking and compute the error with respect to y_mem_result.
        """
        hardwareOn = self.hardwareOn
        scalingOn = self.scalingOn
        
        # Compute benchmark MVM and truncate to match matrix dimensions 
        self.parallelMatVec(type="benchmark", correction=correction)
        if self.y_mem_result is not None:
            self.error = self.y_mem_result - self.y_benchmark_result
            print("\nElement-wise Error: \n", self.error)

            # Compute relative L-infinity norm error
            relative_error = (np.linalg.norm(self.error, ord=np.inf) /
                              np.linalg.norm(self.y_mem_result, ord=np.inf))
            print(f"Relative Loo-norm Error: {relative_error}\n")

            # Store the computed relative L-infinity norm error
            with open(REPORT_PATH, "a+") as file:
                file.write(f"Relative Loo-norm Error: {relative_error}\n")

    def acquireMCAStats(self)-> None:
        """
        Acquires statistics from the MCA object.
        """
        self.mca.getMCAStats()
        return None

    def finalize(self)-> None:
        """
        Finalizes the computation by broadcasting a termination signal to all processes.
        """
        data = np.array([-1, -1], dtype=np.float64)
        self.comm.Bcast(data, root=self.mca.ROOT_PROCESS_RANK)
        return None
