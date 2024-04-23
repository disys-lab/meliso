from .BaseMCA import BaseMCA
import numpy as np
import os,sys,yaml
import meliso

class NonRootMCA(BaseMCA):
    def __init__(self,comm):
        super().__init__(comm)

        # acquire from root process
        self.A = None
        self.localx = None
        self.y = None
        self.startCol = 0
        self.endCol = 0
        self.startRow = 0
        self.endRow = 0
        self.turnOnHardware = 1
        self.mcaStats = np.zeros((self.num_mca_stats,1),dtype=float)

        self.MAX_TOL = 1.0
        self.MIN_TOL = 0.0

        self.device_type = 1

        if "DT" in os.environ.keys():
            self.device_type = int(os.environ["DT"])
        print("INFO: Device type set to {} on rank {}".format(self.device_type,self.rank))

        # compute based on provided params
        self.locRows = 0
        self.locCols = 0

        self.my_row_rank = 0
        self.my_col_rank = 0

        self.device_config = None
        self.getDeviceConfig()

        if "turnOnHardware" in self.exp_config["exp_params"].keys():
            self.turnOnHardware = self.exp_config["exp_params"]["turnOnHardware"]

        self.meliso_obj = meliso.MelisoPy(self.device_type, self.cellRows, self.cellCols, self.MAX_TOL, self.MIN_TOL, self.turnOnHardware)

        if "device_config" not in self.exp_config.keys():
            raise Exception("ExperimentConfigFileError: Device config not specified in %s for MCA rank %s".format(expConfigFile,self.rank))

        if self.device_type == 1:
            self.setConductanceProperties()
            self.setWriteProperties()
            self.setDeviceVariation()
        else:
            print("Device type ={}, ignoring device parameters".format(self.device_type))

        self.comm.Barrier()

        self.setMat()

    def setMat(self):
        self.acquireLocalA()
        self.initializeMCA()

    def acquireLocalA(self):
        decomp_folder_name = self.getDecompositionDir()
        i = int(self.rank/self.mcaRows)
        j = self.rank-(i*self.mcaRows)
        mat_file_path = os.path.join(decomp_folder_name, "{}_{}.npy".format(i, j))
        self.A = np.load(mat_file_path)
        self.locRows = self.A.shape[0]
        self.locCols = self.A.shape[1]

    def initializeMCA(self):
        self.meliso_obj.initializeWeights()
        self.setWeights(self.A)

    def setWeights(self,A):
        self.meliso_obj.setWeights(A)

    def parseRankList(self, rank_list):
        for ranks in rank_list:
            if self.rank in range(ranks[0], ranks[-1]):
                return True
        return False

    def getMCAStats(self):
        recvbuf = None
        self.mcaStats = self.melisoObj.getMCAStats(self.num_mca_stats)
        comm.Gather(mcaStats, recvbuf, root=self.ROOT_PROCESS_RANK)


    def getDeviceConfig(self):
        self.device_config = None
        for device_config_file in self.exp_config["device_config"]["assignment"].keys():
            rank_list = list(self.exp_config["device_config"]["assignment"][device_config_file])
            if int(rank_list[0][0]) == -1 or self.parseRankList(rank_list):
                self.readDeviceConfigFile(device_config_file)

        if self.device_config is None:
            raise Exception(
                "ExperimentConfigFileError: Device config not found/specified for rank {}".format(self.rank))

    def readDeviceConfigFile(self, deviceConfigFile):
        device_config_path = os.path.join(self.exp_config["device_config"]["root"], deviceConfigFile)
        try:
            with open(device_config_path, "r") as stream:
                dc = yaml.safe_load(stream)
                self.device_config = dc["Device"]
        except:
            raise Exception("DeviceConfigFileError: Could not open device config file %s".format(device_config_path))

    def setDeviceVariation(self):
        # NL_LTP, NL_LTD,sigmaDtoD,sigmaCtoC
        NL_LTP = float(self.device_config["DeviceVariation"]["NL_LTP"])
        NL_LTD = float(self.device_config["DeviceVariation"]["NL_LTD"])
        sigmaDtoD = float(self.device_config["DeviceVariation"]["sigmaDtoD"])
        sigmaCtoC = float(self.device_config["DeviceVariation"]["sigmaCtoC"])

        self.meliso_obj.setDeviceVariation(NL_LTP, NL_LTD, sigmaCtoC, sigmaDtoD)

    def setWriteProperties(self):
        writeVoltageLTP = float(self.device_config["WriteProperties"]["writeVoltageLTP"])
        writeVoltageLTD = float(self.device_config["WriteProperties"]["writeVoltageLTD"])
        writePulseWidthLTP = float(self.device_config["WriteProperties"]["writePulseWidthLTP"])
        writePulseWidthLTD = float(self.device_config["WriteProperties"]["writePulseWidthLTD"])
        maxNumLevelLTP = float(self.device_config["WriteProperties"]["maxNumLevelLTP"])
        maxNumLevelLTD = float(self.device_config["WriteProperties"]["maxNumLevelLTD"])
        self.meliso_obj.setWriteProperties(writeVoltageLTP, writeVoltageLTD, writePulseWidthLTP, writePulseWidthLTD,
                                           maxNumLevelLTP, maxNumLevelLTD)

    def setConductanceProperties(self):
        maxConductance = self.device_config["ConductanceProperties"]["maxConductance"]
        minConductance = self.device_config["ConductanceProperties"]["minConductance"]
        avgMaxConductance = maxConductance
        avgMinConductance = minConductance
        conductance = minConductance
        conductancePrev = minConductance
        if "avgMaxConductance" in self.device_config["ConductanceProperties"].keys():  # ["avgMaxConductance"]
            avmc = self.device_config["ConductanceProperties"]["avgMaxConductance"]
            if avmc == "maxConductance":
                avgMaxConductance = maxConductance
        if "avgMinConductance" in self.device_config["ConductanceProperties"].keys():  # ["avgMaxConductance"]
            avmc = self.device_config["ConductanceProperties"]["avgMinConductance"]
            if avmc == "minConductance":
                avgMinConductance = minConductance
        if "conductance" in self.device_config["ConductanceProperties"].keys():
            c = self.device_config["ConductanceProperties"]["conductance"]
            if c == "minConductance":
                conductance = minConductance
        if "conductancePrev" in self.device_config["ConductanceProperties"].keys():
            cp = self.device_config["ConductanceProperties"]["conductancePrev"]
            if cp == "conductancePrev":
                conductancePrev = cp

        self.meliso_obj.setConductanceProperties(maxConductance, minConductance, avgMaxConductance, avgMinConductance,
                                                 conductance, conductancePrev)

    def localMatVec(self, x):
        self.localx = np.copy(x)
        self.meliso_obj.loadInput(x)
        self.meliso_obj.matVec()
        self.y = self.meliso_obj.getResults()

    def parallelMatVec(self):
        x = np.empty(self.locCols, dtype=np.float64)
        self.comm.Recv(x, source=self.ROOT_PROCESS_RANK)
        self.localMatVec(x)
        self.comm.Send(self.y, dest=self.ROOT_PROCESS_RANK)
        return self.y