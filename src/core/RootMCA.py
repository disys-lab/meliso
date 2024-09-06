from .BaseMCA import BaseMCA
from scipy.io import mmread
import numpy as np
import os,sys

class RootMCA(BaseMCA):
    def __init__(self,comm):
        super().__init__(comm)

        self.col_parts = {}
        #dict containing something like rank:[start_col,end_col]
                                        # eg:
                                        # 0 : [0,2]
                                        # 1 : [3,5]
        self.row_parts_ranks = {}
        # dict containing something like {start_row:[ranks that have this row decomposition]}
                                    #eg: [
                                      #   0:[0,1,2,3,4,5],
                                      #   4:[6,7,8,9,10,11]
                                      # ]

        self.matRows = 0
        self.matCols = 0
        self.origMatRows=0
        self.origMatCols=0
        self.mat = None
        self.x = None

        self.mat_min = None
        self.mat_max = None
        self.mat_row_sum = None

        self.x_max = None
        self.x_min = None
        self.x_sum = None

        self.allMCAStats = np.zeros((self.size,self.num_mca_stats,1),dtype=float)

        #self.initializeMatrix(mat)

        # if set_mat:
        #     self.setMat(self.mat)
    
    def verifyConfiguration(self):
        print("Experiment Configuration")
        print(self.exp_config["exp_params"])


    def initializeMatrix(self,mat):
        if mat is None:
            self.matrix_file = None
            
            self.verifyConfiguration()
            if "matrix_file" not in self.exp_config["exp_params"].keys():
                raise Exception("ExperimentConfigFileError: Matrix file not specified in %s".format(self.expConfigFile))

            self.matrix_file = self.exp_config["exp_params"]["matrix_file"]
            self.processMatrixFile()

        else:
            self.mat = mat
            # capture original rows and cols
            self.origMatRows = mat.shape[0]
            self.origMatCols = mat.shape[1]
            self.matRows = mat.shape[0]
            self.matCols = mat.shape[1]

        self.mat, self.mat_min, self.mat_max, self.mat_row_sum = self.scaleMatrix(self.mat)

    def processMatrixFile(self):
        #read the matrix from file
        self.readMatrix(self.matrix_file)

    def readMatrix(self,filename):
        if not os.path.isfile(filename):
            raise Exception("MatrixFileNotFoundError:The file %s does not exist or is invalid".format(filename))
        #0.5*np.eye(66,) #
        mat = mmread(filename)

        if not isinstance(mat, np.ndarray):
            mat = mat.toarray()

        #preprocess and set the matrix
        #self.setMat(mat)

        #capture original rows and cols
        self.origMatRows = mat.shape[0]
        self.origMatCols = mat.shape[1]
        self.matRows = mat.shape[0]
        self.matCols = mat.shape[1]
        self.mat = mat

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
            print(
                "WARNING: The X vector rows {} are not the same as Original Matrix Columns {}".format(
                    x_row,self.origMatCols))

        if x_row > self.matCols:
            raise Exception ("The X vector rows {} exceed Padded Matrix Columns {}".format(
                    x_row,self.matCols))

        cols = self.origMatCols
        mca_cols = self.mcaCols
        cell_cols = self.cellCols

        if mca_cols * cell_cols - cols < 0:
            raise Exception(
                "MatrixMCADimensionMismatch: Cannot encode Matrix (MCA cols {} x Cell cols {}) - Matrix cols {} < 0".format(
                    mca_cols, cell_cols, cols))
        if abs(mca_cols * cell_cols - cols) != cell_cols:
            col_padding_size = abs(mca_cols * cell_cols - cols)
            col_padding = np.zeros((col_padding_size,1), dtype=float)

            print("Adding zero col padding of size ({},{})".format(col_padding_size,1))
            x = np.concatenate([x, col_padding], axis=0)

        if x.shape[0] != self.matCols:
            raise Exception("The Padded X vector rows {} exceed Padded Matrix Columns {}".format(
                x.shape[0], self.matCols))

        self.x = x #,self.x_min,self.x_max = self.scaleMatrix(x)

    def createDecompositionDir(self):
        decomp_folder_name = self.getDecompositionDir()
        if not os.path.isdir(decomp_folder_name):
            os.makedirs(decomp_folder_name, exist_ok=True)

    def scaleMatrix(self,mat):
        #mat = np.copy(matrix)
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
        comm.Gather(mcaStats, self.allMCAStats, root=self.ROOT_PROCESS_RANK)
        
        #self.allMCAStats = allMCAStats

    def globalMatVec(self):
        return None