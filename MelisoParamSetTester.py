import meliso
import numpy as np
from src.core.RootMCA import RootMCA
from src.core.NonRootMCA import NonRootMCA

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
meliso_obj = meliso.MelisoPy(1,16,16,MAX_TOL,MIN_TOL,turnOnHardware)
# meliso_obj.setConductanceProperties(1.2345679012345678e-05, 2.4592986080369874e-07,3.846153846e-8,3.076923077e-9,3.076923077e-9,3.076923077e-9)
# meliso_obj.setWriteProperties(3.2, -2.8, 300e-6,300e-6,16,64)
# meliso_obj.setDeviceVariation(2.4,-4.88,0,0.035)

#Epiram
meliso_obj.setConductanceProperties(1.2345679012345678e-05, 2.4592986080369874e-07,1.2345679012345678e-05, 2.4592986080369874e-07,2.4592986080369874e-07,2.4592986080369874e-07)
meliso_obj.setWriteProperties(5.0, -3.0, 5e-6,5e-6,256,256)
meliso_obj.setDeviceVariation(0.5,-0.5,0,0.02)

setCP = meliso_obj.getConductanceProperties(0,0)
setWP = meliso_obj.getWriteProperties(0,0)
setDV = meliso_obj.getDeviceVariation(0,0)

print(setCP)
print(setWP)
print(setDV)

#
# scaled_A = np.loadtxt(fname='matrices/bcsstk02.mtx',delimiter=',')
# x = np.loadtxt(fname='input_x',delimiter=',')
# #scaled_A = np.random.randint(0,10000,size=(32,32))/10000.0
#
#
# print(scaled_A)
#
# #initialize weights to 0 on the memristor device
# meliso_obj.initializeWeights()
#
# #set weights on the memristor
# meliso_obj.setWeights(scaled_A)
# j=0
# while j<2:
#     #consider the actual input vector and get hardware matvec results
#     #x = np.ones((32,1))
#     #x = np.random.rand(32,1)
#     meliso_obj.loadInput(x)
#     meliso_obj.matVec()
#     y_rescaled_mem_result = meliso_obj.getResults()
#
#     real_Ax = np.dot(scaled_A,x)
#     print("y_rescaled:",y_rescaled_mem_result.reshape((1,32)))
#     print("real_Ax:",real_Ax.reshape((1,32)))
#     print(y_rescaled_mem_result.reshape((1,32))-real_Ax.reshape((1,32)))
#     j = j+1
