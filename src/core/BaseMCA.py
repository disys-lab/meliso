import meliso
import numpy as np
import sys,os,yaml,time


'''
initialize memristor: the first argument is the device type
0: IdealDevice
1: RealDevice
2: MeasuredDevice
3: SRAM
4: DigitalNVM
5: HybridCell
6: _2T1F
For more information read src/cython/Meliso.cpp

Second and third arguments are rows and columns of weight matrix
'''

class BaseMCA:
    def __init__(self,comm):

        #initialize MPI
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        self.num_mca_stats = 8

        self.useMPI4MatDist = True


        if "EXP_CONFIG_FILE" not in os.environ.keys():
            if self.rank == 0:
                print("EXP_CONFIG_FILE not set in os.environ, set it using export EXP_CONFIG_FILE=<path/to/exp/config/file>\n")
            sys.exit(1)

        self.expConfigFile = os.environ["EXP_CONFIG_FILE"]

        #acquire from experiment config file
        self.ROOT_PROCESS_RANK = self.size-1
        self.exp_config =None
        self.decomposition_dir = "/tmp/"
        self.distributed = 0
        self.mcaRows = 1
        self.mcaCols = 1

        # acquire from device config file
        self.cellRows = 1
        self.cellCols = 1

        self.readExpConfig(self.expConfigFile)

        if self.size-1 != self.mcaRows*self.mcaCols:
            raise Exception("ExperimentConfigFileError: MCA grid size {}x{} != mpi processes {}".format(self.mcaRows,self.mcaCols,self.size))

        if "matrix_name" not in self.exp_config["exp_params"].keys():
            raise Exception("ExperimentConfigFileError: Matrix name not specified in %s".format(expConfigFile))

        self.matrix_name = self.exp_config["exp_params"]["matrix_name"]


    def getDecompositionDir(self):
        decomp_id = "{}-{}_{}_{}_{}".format(self.matrix_name,self.mcaRows,self.mcaCols,self.cellRows,self.cellCols)
        decomp_folder_name = os.path.join(self.decomposition_dir,decomp_id)
        return decomp_folder_name

    def getMCAStats(self):
        pass

    def readExpConfig(self,expConfigFile):
        try:
            with open(expConfigFile, "r") as stream:
                self.exp_config = yaml.safe_load(stream)
        except:
            raise Exception("ExperimentConfigFileError: Could not open model file %s".format(expConfigFile))

        if "cell_rows" in self.exp_config["device_config"].keys():
            self.cellRows = self.exp_config["device_config"]["cell_rows"]
        else:
            print("WARNING: Going with default cell rows of {}".format(self.cellRows))

        if "cell_cols" in self.exp_config["device_config"].keys():
            self.cellCols = self.exp_config["device_config"]["cell_cols"]
        else:
            print("WARNING: Going with default cell cols of {}".format(self.cellCols))

        if "distributed" in self.exp_config["exp_params"].keys():
            self.distributed = 1

            if "decomposition_dir" not in self.exp_config["exp_params"]["distributed"].keys():
                print(
                    "ExperimentConfigFileWarning: decomposition_dir not found/specified, using /tmp/ as default")
            else:

                self.decomposition_dir = self.exp_config["exp_params"]["distributed"]["decomposition_dir"]

            if "mca_rows" not in self.exp_config["exp_params"]["distributed"].keys():
                raise Exception(
                    "ExperimentConfigFileError: mcaRows not found/specified in config file {}".format(expConfigFile))
            self.mcaRows = int(self.exp_config["exp_params"]["distributed"]["mca_rows"])

            if "mca_cols" not in self.exp_config["exp_params"]["distributed"].keys():
                raise Exception(
                    "ExperimentConfigFileError: mcaCols not found/specified in config file {}".format(expConfigFile))
            self.mcaCols = int(self.exp_config["exp_params"]["distributed"]["mca_cols"])

