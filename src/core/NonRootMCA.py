from .BaseMCA import BaseMCA
import numpy as np
import os,sys,yaml
import meliso

class NonRootMCA(BaseMCA):
    def __init__(self,comm,HW=-1,SC=-1,set_mat=True):
        super().__init__(comm)

        # acquire from root process
        self.A = None
        self.localx = None
        self.y = None
        self.startCol = 0
        self.endCol = 0
        self.startRow = 0
        self.endRow = 0
        self.turnOnHardware = 0
        self.turnOnScaling = 0
        self.mcaStats = np.zeros((self.num_mca_stats,1),dtype=float)

        self.MAX_TOL = 1.0
        self.MIN_TOL = 0.0
        self.OLIM = 1

        self.PRECISION = 1e-6
        self.ITER_LIMIT = 1000
        self.RESIDUALS_TOL = 1e-6

        self.device_type = 1
        self.interpolants = 3

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

        if HW==-1:
            if "turnOnHardware" in self.exp_config["exp_params"].keys():
                self.turnOnHardware = self.exp_config["exp_params"]["turnOnHardware"]
        else:
            self.turnOnHardware = HW
        if SC==-1:
            if "turnOnScaling" in self.exp_config["exp_params"].keys():
                self.turnOnScaling = self.exp_config["exp_params"]["turnOnScaling"]
        else:
            self.turnOnScaling = SC


        self.meliso_obj = meliso.MelisoPy(self.device_type, self.cellRows, self.cellCols, self.MAX_TOL, self.MIN_TOL, self.turnOnHardware,self.turnOnScaling)

        if "interpolants" in self.exp_config["exp_params"].keys():
            self.interpolants = self.exp_config["exp_params"]["interpolants"]
            print("setting interpolants to {}".format(self.interpolants))

        self.meliso_obj.setInterpolants(self.interpolants)

        if "device_config" not in self.exp_config.keys():
            raise Exception("ExperimentConfigFileError: Device config not specified in %s for MCA rank %s".format(expConfigFile,self.rank))

        if self.device_type == 1:
            self.setConductanceProperties()
            self.setWriteProperties()
            self.setDeviceVariation()

        else:
            print("Device type ={}, ignoring device parameters".format(self.device_type))

        if set_mat:
            self.setMat()

    def setMat(self):
        if not self.useMPI4MatDist:
            self.comm.Barrier()
        self.acquireLocalA()
        self.initializeMCA()

    def acquireLocalA(self):
        if self.useMPI4MatDist:

            self.A = np.zeros((self.cellRows,self.cellCols), dtype=np.float64)
            #print("RANK{}: trying to recieve submatrix".format(self.rank))
            self.comm.Recv(self.A, source=self.ROOT_PROCESS_RANK)
            #print("RANK{}: recieved submatrix".format(self.rank))
        else:
            decomp_folder_name = self.getDecompositionDir()
            i = int(self.rank/self.mcaCols)
            j = self.rank-(i*self.mcaCols)
            mat_file_path = os.path.join(decomp_folder_name, "{}_{}.npy".format(i, j))
            self.A = np.load(mat_file_path)
        self.locRows = self.A.shape[0]
        self.locCols = self.A.shape[1]

    def acquireLocalX(self,x):
        xT = x.reshape((1,self.locCols))
        self.X = np.tile(xT,(self.locRows,1))

    def initializeMCA(self):
        self.meliso_obj.initializeWeights()
        
        try:
            methods = int(os.getenv("WEIGHT_INIT_METHOD"))
        except (TypeError, ValueError):
            raise ValueError("WeightInitMethodError: WEIGHT_INIT_METHOD must be an integer.")
            
        if methods == 0:
            self.setWeightsIncremental(self.A)
        elif methods == 1:
            self.setWeights(self.A)
        else:
            raise ValueError(f"WeightInitMethodError: Only accept 0 (IncrementalInit) or 1 (SingleInit), got {methods}.")


    def setWeights(self,A):
        self.meliso_obj.setWeights(A)

    def setWeightsIncremental(self, A):
        j = 0
        residuals = 0
        while j < self.ITER_LIMIT:
            self.meliso_obj.setWeightsIncremental(A, self.PRECISION)
            actualWeights = self.meliso_obj.getWeights()

            current_residuals = np.linalg.norm(actualWeights - A)
            if abs(residuals - current_residuals)< self.RESIDUALS_TOL and j>0:
                break
            residuals = current_residuals
            j += 1
        print(f"No. iterations: {j}. Current residuals: {residuals}")


    def parseRankList(self, rank_list):
        for ranks in rank_list:
            if self.rank in range(ranks[0], ranks[-1]):
                return True
        return False

    def getMCAStats(self):
        recvbuf = None
        self.mcaStats = self.melisoObj.getMCAStats(self.num_mca_stats)
        self.comm.Gather(self.mcaStats, recvbuf, root=self.ROOT_PROCESS_RANK)

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
        maxConductance = float(self.device_config["ConductanceProperties"]["maxConductance"])
        minConductance = float(self.device_config["ConductanceProperties"]["minConductance"])
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

    def denoiseLeastSquare(self, w, lbda=1e-6):
        rows = w.shape[0]
        I = np.eye(rows)
        L = np.eye(rows)
        for i in range(rows - 1):
            L[i, i + 1] = -1

        print("Applying least-square denoising...")
        y = np.linalg.inv(I + lbda * L @ L.T) @ w
        return y

    def localMatVec(self, x, RESULT_MULT=1.0):
        print(f"Computing MVM at Device Rank {self.rank}...")
        self.meliso_obj.loadInput(x)
        self.meliso_obj.matVec()
        y = RESULT_MULT * self.meliso_obj.getResults()
        return y

    def parallelMatVec(self):
        self.y = self.errorCorrection()
        self.comm.Send(self.y, dest=self.ROOT_PROCESS_RANK)
        return self.y
    
    def errorCorrection(self):
        x = np.empty(self.locCols, dtype=np.float64)
        self.comm.Recv(x, source=self.ROOT_PROCESS_RANK)
        self.acquireLocalX(x)

        y_a = np.zeros((self.locRows, 1), dtype=float)
        
        # Compute U_tilde = A @ x_tilde:
        U_tilde = np.empty((self.locRows, 1), dtype=float)
        X_tilde = np.copy(self.X)
        A_tilde = np.copy(self.A)

        self.meliso_obj.initializeWeights()
        self.setWeightsIncremental(X_tilde)
        X_tilde = self.meliso_obj.getWeights()
        
        for i in range(self.locRows):
            ai = self.A[i, :].flatten()
            ui_tilde = self.localMatVec(ai).flatten()
            ui_tilde = self.denoiseLeastSquare(ui_tilde)
            U_tilde[i] = ui_tilde[i]

        # Compute V_tilde = A_tilde @ x:
        V_tilde_a = np.empty((self.locRows, 1), dtype=float)
        self.meliso_obj.initializeWeights()
        self.setWeightsIncremental(A_tilde)
        A_tilde = self.meliso_obj.getWeights()
        for i in range(self.locRows):
            xT_i = X_tilde[i,:].flatten()
            v_tilde_i = self.localMatVec(xT_i)
            v_tilde_i = self.denoiseLeastSquare(v_tilde_i)
            V_tilde_a[i] = v_tilde_i[i]

        # Compute y_tilde:
        y_tilde = self.localMatVec(x)
        y_tilde = self.denoiseLeastSquare(y_tilde)

        for i in range(self.locRows):
            y_a[i] = y_tilde[i] - V_tilde_a[i] + U_tilde[i]

        y_corr = y_a
        return y_corr