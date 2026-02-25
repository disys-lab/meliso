# ==================================================================================================
# @author: Huynh Quang Nguyen Vo, Paritosh Ramanan
# @affiliation: Oklahoma State University
# @date: 2026-02-24
# ==================================================================================================
import meliso
import sys, os, yaml


"""
Initialize device materials for memristor crossbar arrays.

The first argument is the device type. The supported device types are:
    0: IdealDevice
    1: RealDevice
    2: MeasuredDevice
    3: SRAM
    4: DigitalNVM
    5: HybridCell
    6: _2T1F
For more information read src/cython/Meliso.cpp

Second and third arguments are rows and columns of weight matrix.
"""

class BaseMCA:
    def __init__(self,comm):

        # MPI Handler
        self.comm = comm
        self.rank = self.comm.Get_rank()
        self.size = self.comm.Get_size()

        # --- Number of MCA stats to be collected ---
        # (1) totalSubArrayArea; (2) totalNeuronAreaIH; (3) subArrayIHLeakage; (4) leakageNeuronIH
        # (5) subArrayIH->writeLatency
        # (6) arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy
        # (7) subArrayIH->readLatency
        # (8) arrayIH->readEnergy + subArrayIH->readDynamicEnergy
        self.num_mca_stats = 8

        self.useMPI4MatDist = True

        # --- Error correction scheme ---
        # 0: no error correction; 1: with error correction
        # Default is 1 (write-and-verify -> first-order -> second-order error correction)
        self.ERR_CORR = 1


        if "EXP_CONFIG_FILE" not in os.environ.keys():
            if self.rank == 0:
                print("[ERROR] EXP_CONFIG_FILE not set in the environmental variable\n")
                print("Set it using the command: `export EXP_CONFIG_FILE=<path/to/exp/config/file>`\n")
            sys.exit(1)

        self.expConfigFile = os.environ["EXP_CONFIG_FILE"]

        # --- Acquire from experiment config file ---
        # The file should be in YAML format. 
        # For more information, read the sample file at <home>/config_files/quickstart/example.yaml
        self.ROOT_PROCESS_RANK = self.size - 1
        self.exp_config = None
        self.decomposition_dir = os.environ.get("TMPDIR", "/tmp/")
        self.distributed = 0
        self.mcaRows = 1    # Number of MCAs in the horizontal direction
        self.mcaCols = 1    # Number of MCAs in the vertical direction
        self.cellRows = 1   # Number of rows in each MCA cell
        self.cellCols = 1   # Number of columns in each MCA cell

        self.readExpConfig(self.expConfigFile)

        assert self.exp_config is not None, "ExperimentConfigFileError: Could not read experiment config file {}".format(self.expConfigFile)

        # --- Handling the error correction logic ---
        # Prefer environmental variable override; otherwise honor YAML; otherwise keep default.
        if "EC" in os.environ.keys():
            self.ERR_CORR = int(os.environ["EC"])
        if "errCorr" in self.exp_config["exp_params"].keys():
            self.ERR_CORR = self.exp_config["exp_params"]["errCorr"]

        # Check whether the MCA grid size matches the number of MPI processes
        if self.size - 1 != self.mcaRows*self.mcaCols:
            raise Exception(f"ExperimentConfigFileError: MCA grid size {self.mcaRows}x{self.mcaCols} != mpi processes {self.size}")

        # Check whether the matrix name is specified in the experiment config file
        if "matrix_name" not in self.exp_config["exp_params"].keys():
            raise Exception(f"ExperimentConfigFileError: Matrix name not specified in {self.expConfigFile}")
        self.matrix_name = self.exp_config["exp_params"]["matrix_name"]


    def getDecompositionDir(self):
        """ 
        Get the directory for storing the MCA decomposition results. 
        The directory name is based on the matrix name and the MCA configuration. 
        """
        decomp_id = f"{self.matrix_name} - {self.mcaRows}x{self.mcaCols}x{self.cellRows}x{self.cellCols}"
        decomp_folder_name = os.path.join(self.decomposition_dir, decomp_id)
        return decomp_folder_name

    def getMCAStats(self):
        """
        Get the MCA statistics collected during the experiment. 
        The statistics are stored in a list of length `self.num_mca_stats`. 
        """
        pass

    def readExpConfig(self, expConfigFile):
        """
        Read the experiment configuration file. 
        The configuration file should be in YAML format, and should be located at the path specified 
        by the `EXP_CONFIG_FILE` environmental variable.
        """
        try:
            with open(expConfigFile, "r") as stream:
                self.exp_config = yaml.safe_load(stream)
        except:
            raise Exception(f"ExperimentConfigFileError: Could not open model file {expConfigFile}")

        if "cell_rows" in self.exp_config["device_config"].keys():
            self.cellRows = self.exp_config["device_config"]["cell_rows"]
        else:
            print(f"[WARNING] Going with default cell rows of {self.cellRows}")

        if "cell_cols" in self.exp_config["device_config"].keys():
            self.cellCols = self.exp_config["device_config"]["cell_cols"]
        else:
            print(f"[WARNING] Going with default cell cols of {self.cellCols}")

        if "distributed" in self.exp_config["exp_params"].keys():
            self.distributed = 1
            # prefer env override; otherwise honor YAML; otherwise keep default.
            env_decomp = os.environ.get("TMPDIR")
            if env_decomp:
                self.decomposition_dir = env_decomp
            elif "decomposition_dir" in self.exp_config["exp_params"]["distributed"].keys():
                self.decomposition_dir = self.exp_config["exp_params"]["distributed"]["decomposition_dir"]
            else:
                print(f"[WARNING] decomposition_dir not found; using {self.decomposition_dir}")

            if "mca_rows" not in self.exp_config["exp_params"]["distributed"].keys():
                raise Exception(
                    f"ExperimentConfigFileError: mcaRows not found/specified in config file {expConfigFile}")
            self.mcaRows = int(self.exp_config["exp_params"]["distributed"]["mca_rows"])

            if "mca_cols" not in self.exp_config["exp_params"]["distributed"].keys():
                raise Exception(
                    f"ExperimentConfigFileError: mcaCols not found/specified in config file {expConfigFile}")
            self.mcaCols = int(self.exp_config["exp_params"]["distributed"]["mca_cols"])

