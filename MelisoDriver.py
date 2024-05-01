import numpy as np
from scipy.io import mmread
from scipy.sparse import csr_matrix
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
DEVICE_TYPE = 1
DIM = 66
MAX_TOL = 1.0
MIN_TOL = 1e-6
TURN_ON_HARDWARE = 1
TURN_ON_SCALING = 1
RANDOM_SEED = 28
MATRIX_PATH = 'matrices/bcsstk02.mtx'

# Initialize Meliso object
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

# Initialize weights on the memristor device
memristor.initializeWeights()
memristor.setWeights(scaled_A)

# Single-run experiment
memristor.loadInput(x)
memristor.matVec()
y_mca = memristor.getResults().reshape((1, DIM))
y = np.dot(scaled_A, x).reshape((1, DIM))
print("y_MCA:", y_mca)
print("y:", y)
print("Error:", y_mca - y)