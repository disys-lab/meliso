import os
import numpy as np
from pathlib import Path
from .BaseMCA import BaseMCA
from scipy.io import mmread


#===================================================================================================
# Utility functions
#===================================================================================================
def __report_path__(fallback_name: str = "default_report.txt") -> Path:
    env_path = os.environ.get("REPORT_PATH")
    
    if env_path:    # Use the exact path provided by `REPORT_PATH` environment variable if it exists
        path = Path(env_path)
    else:           # Fallback for manual testing outside of the bash script
        path = Path(fallback_name)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path

def __write_report__(content: str):
    path = __report_path__()
    
    with path.open('a+') as f:
        f.write(content + '\n')
        
    return None

#===================================================================================================
# CLASS DEFINITION
#===================================================================================================
class RootMCA(BaseMCA):
    def __init__(self,comm):
        super().__init__(comm)

        # Dictionary containing the column start and end indices for each rank's matrix chunk.
        # This is used to determine which portion of the input vector `x`` to send to each rank during the parallel MVM.
        # Dictionary format: {rank: [col_start_index, col_end_index]}
        self.col_parts = {}

        # Dictionary containing the row start indices and the corresponding ranks that have that row decomposition.
        # This is used to determine which ranks to receive the partial results from during the parallel MVM and how to sum them up to get the final output vector `y`.
        # Dictionary format: {row_start_index: [ranks that have this row decomposition]}
        self.row_parts_ranks = {}

        self.matRows = 0
        self.matCols = 0
        self.origMatRows=0
        self.origMatCols=0
        self.mat = None
        self.globalMat = None
        self.x = None
        self.globalX = None

        self.mat_min = None
        self.mat_max = None
        self.mat_row_sum = None

        self.x_max = None
        self.x_min = None
        self.x_sum = None

        self.allMCAStats = np.zeros((self.size,self.num_mca_stats,1),dtype=float)
    
    def printConfiguration(self):
        """
        Display the current experimental settings for an instance of the `RootMCA` class. 
        This includes all parameters defined in the experiment configuration file in.
        The configuration file should be located at `config_files/<experiment_name>/<material_name>/` directory.
        """
        assert self.exp_config is not None, \
            "ExperimentConfigError: Experiment configuration is not set for this RootMCA instance."
        print("\nExperiment Configuration")
        print(self.exp_config["exp_params"])

        # Write the configuration to the report file
        string_builder = []
        string_builder.append("\nExperiment Configuration")
        string_builder.append(str(self.exp_config["exp_params"]))
        __write_report__("\n".join(string_builder))


    def initializeMatrix(self,mat):
        """Initialize the matrix for the MVM operation."""
        if mat is None:
            self.matrix_file = None
            
            self.printConfiguration()
            if self.exp_config is not None:
                if "matrix_file" not in self.exp_config["exp_params"].keys():
                    raise Exception(f"ExperimentConfigFileError: Matrix file not specified in {self.expConfigFile}")
                else:
                    self.matrix_file = self.exp_config["exp_params"]["matrix_file"]
                    self.processMatrixFile()

        else:
            self.mat = mat

            # Capture original rows and cols of the matrix before any scaling or padding is applied
            self.origMatRows = mat.shape[0]
            self.origMatCols = mat.shape[1]
            self.matRows = mat.shape[0]
            self.matCols = mat.shape[1]

        self.mat, self.mat_min, self.mat_max, self.mat_row_sum = self.scaleMatrix(self.mat)

        if self.exp_config is not None:
            self.hardwareOn = self.exp_config["exp_params"]["turnOnHardware"]
            self.scalingOn = self.exp_config["exp_params"]["turnOnScaling"]

    def processMatrixFile(self):
        """Load the external data into the `RootMCA` instance."""
        self.readMatrix(self.matrix_file)

    def readMatrix(self,filename):
        """
        Read the matrix from the specified file and store it in the `RootMCA` instance.
        Supported file formats include .mtx, .npy, .csv, and .txt. The method also captures the 
        original dimensions of the matrix before any scaling or padding is applied.
        """
        if not os.path.isfile(filename):
            raise Exception(f"MatrixFileNotFoundError:The file {filename} \
                            does not exist or is invalid")
        
        mat = None
        if filename.endswith('.mtx'):
            mat = mmread(filename)
            if not isinstance(mat, np.ndarray):
                mat = mat.toarray()
        elif filename.endswith('.npy'):
            mat = np.load(filename)
        elif filename.endswith('.csv'):
            mat = np.loadtxt(filename, delimiter=',')
        elif filename.endswith('.txt'):
            mat = np.loadtxt(filename, delimiter=',')
        else:
            raise Exception("MatrixFileFormatError: The file format is not supported. \
                            Current supported formats are .mtx and .npy")

        # Capture original rows and cols of the matrix before any scaling or padding is applied
        self.origMatRows = mat.shape[0]
        self.origMatCols = mat.shape[1]
        self.matRows = mat.shape[0]
        self.matCols = mat.shape[1]
        self.mat = mat
        self.globalMat = mat


    def setMat(self,mat):

        #scale matrix
        #mat = self.scaleMatrix(mat)

        #pad the matrix
        self.mat, self.matRows, self.matCols = self.padMatrix(mat)

        #TODO: Add a check here to see if file based distribution or MPI based Distribution
        # create experiment directory
        self.createDecompositionDir()

        #propagate matrix decomposition
        self.distributeMatrixChunksFileWrite()

        if not self.useMPI4MatDist:
            self.comm.Barrier()

    def setX(self,x):
        x_row = x.shape[0]

        if x_row < self.origMatCols:
            print(f"WARNING: The X vector rows {x_row} are not the same as Original Matrix Columns {self.origMatCols}")

        if x_row > self.matCols:
            raise Exception(f"The X vector rows {x_row} exceed Padded Matrix Columns {self.matCols}")

        cols = self.origMatCols
        mca_cols = self.mcaCols
        cell_cols = self.cellCols

        if mca_cols * cell_cols - cols < 0:
            raise Exception(
                f"MatrixMCADimensionMismatch: Cannot encode Matrix (MCA cols {mca_cols} x Cell cols {cell_cols}) - Matrix cols {cols} < 0")
        if abs(mca_cols * cell_cols - cols) != cell_cols:
            col_padding_size = abs(mca_cols * cell_cols - cols)
            col_padding = np.zeros((col_padding_size,1), dtype=float)

            print(f"Adding zero col padding of size ({col_padding_size},{1})")
            x = np.concatenate([x, col_padding], axis=0)

        if x.shape[0] != self.matCols:
            raise Exception(f"The Padded X vector rows {x.shape[0]} exceed Padded Matrix Columns {self.matCols}")

        self.x = x


    def createDecompositionDir(self):
        decomp_folder_name = self.getDecompositionDir()
        if not os.path.isdir(decomp_folder_name):
            os.makedirs(decomp_folder_name, exist_ok=True)

    def scaleMatrix(self,mat):
        mat = mat.astype(np.float64)
        mat_row_sum = np.sum(mat, axis=1)
        mat_min = mat.min()
        mat -= mat_min
        mat_max = mat.ptp()
        mat /= mat_max
        return mat,mat_min,mat_max,mat_row_sum

    def padMatrix(self,mat):
        rows = self.origMatRows = mat.shape[0]
        cols = self.origMatCols = mat.shape[1]
        mca_rows = self.mcaRows
        mca_cols = self.mcaCols
        cell_rows = self.cellRows
        cell_cols = self.cellCols

        # if mca_rows*cell_rows - rows > 0:
        #     raise Exception("MatrixMCADimensionMismatch: Cannot encode Matrix (MCA rows {} x Cell rows {}) > Matrix rows {}".format(mca_rows,cell_rows,rows))
        if mca_rows*cell_rows - rows < 0:
            raise Exception(
                "MatrixMCADimensionMismatch: Cannot encode Matrix (MCA rows {} x Cell rows {}) - Matrix rows {} < 0".format(
                    mca_rows, cell_rows, rows))

        if abs(mca_rows*cell_rows - rows) != cell_rows:
            row_padding_size = abs(mca_rows*cell_rows - rows)
            row_padding = np.zeros((row_padding_size,cols),dtype=float)
            print("Adding zero row padding of size ({},{})".format(row_padding_size,cols))
            mat = np.concatenate([mat,row_padding],axis=0)
            rows = rows+row_padding_size
        if mca_rows*cell_rows - rows > cell_rows:
            reduction = int((mca_rows*cell_rows - rows)/cell_rows)
            print("WARNING: The Row matrix placement efficiency on MCA Grid is not maximized! You can reduce number of MCA Grid rows by {}".format(reduction))

        # if mca_cols*cell_cols - cols>0:
        #     raise Exception("MatrixMCADimensionMismatch: Cannot encode Matrix (MCA cols {} x Cell cols {}) > Matrix cols {}".format(mca_cols,cell_cols,cols))
        if mca_cols*cell_cols - cols < 0:
            raise Exception(
                "MatrixMCADimensionMismatch: Cannot encode Matrix (MCA cols {} x Cell cols {}) - Matrix cols {} < 0".format(
                    mca_cols, cell_cols, cols))
        if abs(mca_cols*cell_cols - cols) != cell_cols:
            col_padding_size = abs(mca_cols*cell_cols - cols)
            col_padding = np.zeros((rows,col_padding_size),dtype=float)

            print("Adding zero col padding of size ({},{})".format(rows,col_padding_size))
            mat = np.concatenate([mat,col_padding],axis=1)
            cols = cols +col_padding_size

        if mca_cols*cell_cols - cols > cell_cols:
            reduction = int((mca_cols*cell_cols - rows)/cell_cols)
            print("WARNING: The Col matrix placement efficiency on MCA Grid is not maximized! You can reduce number of MCA Grid cols by {}".format(reduction))

        return mat,rows,cols

    def distributeMatrixChunksFileWrite(self):
        self.col_parts = {}
        self.row_parts_ranks = {}
        decomp_folder_name = self.getDecompositionDir()
        if self.distributed:
            for i in range(self.mcaRows):
                for j in range(self.mcaCols):
                    #print("ROOT:  {},{} ".format(self.matRows, self.matCols))
                    sc,ec,sr,er = self.position_assign(self.cellRows,self.cellCols,self.mcaRows,self.mcaCols,self.matRows,self.matCols,i,j)

                    mat_ij = self.mat[sr:er,sc:ec]

                    #print("Rank({},{})--->SR:{},ER:{}\t SC:{},EC:{},matshape:{}".format(i, j, sr, er, sc, ec,mat_ij.shape))

                    mat_file_path = os.path.join(decomp_folder_name,"{}_{}.npy".format(i,j))
                    np.save(mat_file_path,mat_ij)

                    rank = i*self.mcaCols +j

                    if self.useMPI4MatDist:
                        data = np.copy(mat_ij)
                        #print("ROOT: sending submatrix {}{} to {}".format(i,j,rank))
                        self.comm.Send(data,dest=rank)
                        #print("ROOT: sent submatrix to {}".format(rank))

                    if sr not in self.row_parts_ranks.keys():
                        self.row_parts_ranks[sr] = []

                    self.row_parts_ranks[sr].append(rank)

                    if rank not in self.col_parts.keys():
                        self.col_parts[rank] = [sc,ec]

                    #self.col_parts[rank].append([sc,ec])

    # def distributeMatrixChunksMPI(self):
    #
    #     raise Exception("NotImplementedError:Matrix chunks distributing via MPI.Scatter/Gather not implemented yet")

    def position_assign(self, P, Q, M, N, R, C, i, j):
        """
        Simple position assign. Mapping strategy: Matrix (RxC) -> (MP)x(NQ)
        Assumption:
          1. A memristor crossbar array device contains an (MxN) number of chiplets,
             each has an(PxQ) number of cells.
          2. For simplicity, R = C, while R = M*P and C = N * Q.

        (P,Q): Dimension of memristor crossbar array unit.
        (M,N): Dimension of memristor crossbar array chiplet.
        (R,C): Dimension of the matrix.
        """

        assert (R == M * P), "We cannot map this matrix rows {} to the device {}x{}".format(R,M,P)
        assert (C == N * Q), "We cannot map this matrix cols {} to the device {}x{}".format(C,N,Q)

        # Assign the position based on the chiplet's index
        # For example, (i=3,j=3) means the chiplet located at Row 2 and Column 3
        true_row = i
        true_column = j
        if (true_row == 0):
            s_r = 0
            e_r = P
        else:
            s_r = true_row * P
            e_r = true_row * P + P

            # if e_r > M:
            #     e_r = M

        if (true_column == 0):
            s_c = 0
            e_c = Q
        else:
            s_c = true_column * Q
            e_c = true_column * Q + Q

            # if e_c > N:
            #     e_c = N

        return s_c, e_c, s_r, e_r

    def parallelMatVec(self):
        x = self.x #np.copy(self.x)
        # print(self.col_parts)
        # print(self.row_parts_ranks)
        # send chunks of x to all processes on chosen row
        for rank in self.col_parts.keys():
            start = self.col_parts[rank][0]
            end = self.col_parts[rank][1]
            #print(start,end,rank)
            self.comm.Send(x[start:end,:], dest=rank)

        sum_y = np.zeros(self.matRows,dtype=np.float64)

        #print("ROOT: sent x",self.row_parts_ranks)

        for sr in self.row_parts_ranks.keys():
            #row start index
            start = sr

            #col start index
            end = start+self.cellRows

            #list of ranks that have this row decomposition
            rank_list = self.row_parts_ranks[sr]

            #iterate through the ranks
            for rank in rank_list:

                #initialize buffer for each rank
                y = np.empty(end - start, dtype=np.float64)

                #recieve from each rank
                self.comm.Recv(y, source=rank)
                #print("ROOT: recieved y from {}".format(rank),y)
                #print("ROOT: true result must be",np.dot(self.mat,self.x))
                # print(self.mat.shape)
                #add the result to the running sum of that rank.
                sum_y[start:end] = sum_y[start:end] + y

        #print("ROOT: recieved all ys ")

        return sum_y[:self.origMatRows]

    def getMCAStats(self):
        mcaStats = np.zeros((self.num_mca_stats,1),dtype=float)
        self.comm.Gather(mcaStats, self.allMCAStats, root=self.ROOT_PROCESS_RANK)
        for rank in range(self.size):
            print(f"MCAStats for Rank {rank}")
            
            print(f"\t totalSubArrayArea = {self.allMCAStats[rank][0][0]}")
            print(f"\t totalNeuronAreaIH = {self.allMCAStats[rank][1][0]}")
            print(f"\t subArrayIHLeakage = {self.allMCAStats[rank][2][0]}")
            print(f"\t leakageNeuronIH = {self.allMCAStats[rank][3][0]}")

            print(f"\t subArrayIH->writeLatency = {self.allMCAStats[rank][4][0]}")
            print(f"\t arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy = {self.allMCAStats[rank][5][0]}")
            print(f"\t subArrayIH->readLatency = {self.allMCAStats[rank][6][0]}")
            print(f"\t arrayIH->readEnergy + subArrayIH->readDynamicEnergy = {self.allMCAStats[rank][7][0]}")

        self.allMCAStats = self.allMCAStats.reshape((self.size,self.num_mca_stats))

        writeLat = self.allMCAStats[:,4]
        writeEnergy = self.allMCAStats[:,5]

        readLat= self.allMCAStats[:,6]
        readEnergy = self.allMCAStats[:,7]

        print("\nOverall MCA Stats:")
        print(f"EC= {self.ERR_CORR}; writeLatency Mean = {np.mean(writeLat)} [s], stddev = {np.std(writeLat)} [s]")
        print(f"EC= {self.ERR_CORR}; writeLatency Max = {np.max(writeLat)} [s], Min = {np.min(writeLat)} [s]")
        print(f"EC= {self.ERR_CORR}; writeEnergy Mean = {np.mean(writeEnergy)} [J], stddev = {np.std(writeEnergy)} [J]")
        print(f"EC= {self.ERR_CORR}; writeEnergy Max = {np.max(writeEnergy)} [J], Min = {np.min(writeEnergy)} [J]")

        print(f"EC= {self.ERR_CORR}; readLatency Mean = {np.mean(readLat)} [s], stddev = {np.std(readLat)} [s]")
        print(f"EC= {self.ERR_CORR}; readLatency Max = {np.max(readLat)} [s], Min = {np.min(readLat)} [s]")
        print(f"EC= {self.ERR_CORR}; readEnergy Mean = {np.mean(readEnergy)} [J], stddev = {np.std(readEnergy)} [J]")
        print(f"EC= {self.ERR_CORR}; readEnergy Max = {np.max(readEnergy)} [J], Min = {np.min(readEnergy)} [J]")

        # Write the MCA stats to the report file
        string_builder = []
        string_builder.append("\nOverall MCA Stats:")
        string_builder.append(f"EC= {self.ERR_CORR}; writeLatency Mean = {np.mean(writeLat)} [s], stddev = {np.std(writeLat)} [s]")
        string_builder.append(f"EC= {self.ERR_CORR}; writeLatency Max = {np.max(writeLat)} [s], Min = {np.min(writeLat)} [s]")
        string_builder.append(f"EC= {self.ERR_CORR}; writeEnergy Mean = {np.mean(writeEnergy)} [J], stddev = {np.std(writeEnergy)} [J]")
        string_builder.append(f"EC= {self.ERR_CORR}; writeEnergy Max = {np.max(writeEnergy)} [J], Min = {np.min(writeEnergy)} [J]")
        string_builder.append(f"EC= {self.ERR_CORR}; readLatency Mean = {np.mean(readLat)} [s], stddev = {np.std(readLat)} [s]")
        string_builder.append(f"EC= {self.ERR_CORR}; readLatency Max = {np.max(readLat)} [s], Min = {np.min(readLat)} [s]")
        string_builder.append(f"EC= {self.ERR_CORR}; readEnergy Mean = {np.mean(readEnergy)} [J], stddev = {np.std(readEnergy)} [J]")
        string_builder.append(f"EC= {self.ERR_CORR}; readEnergy Max = {np.max(readEnergy)} [J], Min = {np.min(readEnergy)} [J]")
        __write_report__("\n".join(string_builder))