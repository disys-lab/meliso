import meliso
import numpy as np

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
MAX_TOL = 1.0
MIN_TOL = 1e-6
DIM = 32
RANDOM_SEED = 28

# Parameters
DEVICE_TYPE = 1
TURN_ON_HARDWARE = 1
TURN_ON_SCALING = 0

memristor = meliso.MelisoPy(DEVICE_TYPE, DIM, DIM, MAX_TOL, MIN_TOL, TURN_ON_HARDWARE, TURN_ON_SCALING)

# Matrix and vector initialization
np.random.seed(RANDOM_SEED)
A = np.random.randn(DIM, DIM)
x = np.random.randn(DIM, 1)

# Memristor MVM
memristor.initializeWeights()
memristor.setWeights(A)
memristor.loadInput(x)
memristor.matVec()
y_mca = memristor.getResults().reshape((1, DIM))

# Real MVM
y = np.dot(A, x).reshape((1, DIM))

# Difference between Memristor and Real MVM
print(f"L2-norm: {np.linalg.norm(y - y_mca)}\n")
print(f"Loo-norm: {np.linalg.norm(y - y_mca, ord=np.inf)}\n")

with open("output.txt", "a+") as file:
    file.write(f"L2-norm: {np.linalg.norm(y - y_mca)}\n")
    file.write(f"Loo-norm: {np.linalg.norm(y - y_mca, ord=np.inf)}\n")