import meliso
import numpy as np
from scipy.io import mmread

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
NRUNS = 1000
MAX_TOL = 1.0
MIN_TOL = 0.5
dim = 32

turnOnHardware = 1
turnOnScaling = 1
meliso_obj = meliso.MelisoPy(1,dim,dim,MAX_TOL,MIN_TOL,turnOnHardware,turnOnScaling)

RMSE = []
for run in range(NRUNS):
    # Create a standard normal matrix
    scaled_A = np.random.randn(dim, dim)
    print(f"Test Matrix: {scaled_A}")

    # Initialize weights on the memristor device
    meliso_obj.initializeWeights()

    # Set weights on the memristor
    meliso_obj.setWeights(scaled_A)

    j = 0
    while j<1:
        x = np.random.randn(dim, 1)

        # Matrix Multiplication (Ax = y) on MCA:
        meliso_obj.loadInput(x)
        meliso_obj.matVec()
        y_MCA = meliso_obj.getResults()
        print(f"y_MCA: {y_MCA.reshape((1,dim))}")


        # Matrix Multiplication (Ax = y) on Ground Truth:
        y = np.dot(scaled_A,x)
        print(f"y: {y.reshape((1,dim))}")

        # Difference between MCA's and Ground Truth's result:
        rmse = np.sqrt(np.square(np.subtract(y,y_MCA)).mean())
        print(f"Iteration{run+1}: RMSE = {rmse}\n")

        RMSE.append(rmse)
        j +=1

RMSE = np.array([RMSE]); RMSE = RMSE.reshape((NRUNS,1)); np.savetxt("./results/RMSE.txt", RMSE)
