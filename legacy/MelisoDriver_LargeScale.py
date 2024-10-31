import numpy as np
from scipy.io import mmread
import meliso

"""
MelisoPy: Initializes a memristor device with a specified type and dimensions for the weight matrix.

Parameters:
    device_type (int): Specifies the type of memristor device to be initialized. The options are:
        0: IdealDevice - A perfect memristor with ideal behavior.
        1: RealDevice - A realistic memristor with practical imperfections.
        2: MeasuredDevice - A memristor that uses measured data for simulation.
        3: SRAM - Static Random Access Memory type memristor.
        4: DigitalNVM - Digital Non-Volatile Memory memristor.
        5: HybridCell - A hybrid type memristor cell.
        6: _2T1F - A dual transistor, single ferroelectric memristor.
    rows (int): The number of rows in the weight matrix.
    columns (int): The number of columns in the weight matrix.
    MAX_TOL (float)
    MIN_TOL (float)
    TURN_ON_HARDWARE (int)
    TURN_ON_SCALING (int)

Returns:
    memristor (objet)

Notes:
    For more detailed information about each device type, refer to the implementation in src/cython/Meliso.cpp.
"""

# Constants
NRUNS = 1000
MAX_TOL = 1.0
MIN_TOL = 0.5
DIM = 32
RANDOM_SEED = 28
RESULTS_PATH = "./results/"
MATRIX_PATH = ""

# Memristor initialization parameters
DEVICE_TYPE = 1  # RealDevice
TURN_ON_HARDWARE = 1
TURN_ON_SCALING = 1

# Initialize memristor object
memristor = meliso.MelisoPy(DEVICE_TYPE, DIM, DIM, MAX_TOL, MIN_TOL, TURN_ON_HARDWARE, TURN_ON_SCALING)

# Matrix and vector initialization
np.random.seed(RANDOM_SEED)
if not MATRIX_PATH:
    scaled_A = np.random.randn(DIM, DIM)
else:
    scaled_A = mmread(MATRIX_PATH)
    scaled_A = scaled_A.toarray()
    if scaled_A.shape != (DIM,DIM):
        DIM = scaled_A.shape[0]
        memristor = meliso.MelisoPy(DEVICE_TYPE, DIM, DIM, MAX_TOL, MIN_TOL, 
                                    TURN_ON_HARDWARE, TURN_ON_SCALING)
x = np.random.randn(DIM, 1)

# N-runs experiment:
RMSE = []; ERRORS = []
for run in range(NRUNS):
    memristor.initializeWeights()
    memristor.setWeights(scaled_A)
    memristor.loadInput(x)
    memristor.matVec()
    y_mca = memristor.getResults().reshape((1, DIM))

    y = np.dot(scaled_A, x).reshape((1, DIM))
    error = y - y_mca
    rmse = np.sqrt(np.mean(np.square(error)))

    RMSE.append(rmse)
    ERRORS.append(error.flatten())

    print(f"Run: {run + 1}")
    print(f"Error vector: {error}")
    print(f"RMSE = {rmse}")
    print("#" * 20)


# Save results to files
np.savetxt(f"{RESULTS_PATH}RMSE.txt", RMSE)
np.savetxt(f"{RESULTS_PATH}ERRORS.txt", np.hstack(ERRORS))
