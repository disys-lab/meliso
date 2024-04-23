import meliso
import numpy as np

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

MAX_TOL = 1.0
MIN_TOL = 0.0

turnOnHardware = 1
meliso_obj = meliso.MelisoPy(1,32,32,MAX_TOL,MIN_TOL,turnOnHardware)

#obtain an A matrix with values between 0,1
#I have observed that having matrix between 0,1 gives the best results

scaled_A = np.loadtxt(fname='scaled_A',delimiter=',')
x = np.loadtxt(fname='input_x',delimiter=',')
#scaled_A = np.random.randint(0,10000,size=(32,32))/10000.0



print(scaled_A)

#initialize weights to 0 on the memristor device
meliso_obj.initializeWeights()

#set weights on the memristor
meliso_obj.setWeights(scaled_A)
j=0
while j<2:
    #consider the actual input vector and get hardware matvec results
    #x = np.ones((32,1))
    #x = np.random.rand(32,1)
    meliso_obj.loadInput(x)
    meliso_obj.matVec()
    y_rescaled_mem_result = meliso_obj.getResults()

    real_Ax = np.dot(scaled_A,x)
    print("y_rescaled:",y_rescaled_mem_result.reshape((1,32)))
    print("real_Ax:",real_Ax.reshape((1,32)))
    print(y_rescaled_mem_result.reshape((1,32))-real_Ax.reshape((1,32)))
    j = j+1
