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
meliso_obj = meliso.MelisoPy(0,32,32,MAX_TOL,MIN_TOL,turnOnHardware)
meliso_obj.setConductanceProperties(3.846153846e-8, 3.076923077e-9,3.846153846e-8,3.076923077e-9,3.076923077e-9,3.076923077e-9)
meliso_obj.setWriteProperties(3.2, -2.8, 300e-6,300e-6,16,64)
meliso_obj.setDeviceVariation(2.4,-4.88,0,0.035)

setCP = meliso_obj.getConductanceProperties(0,0)
setWP = meliso_obj.getWriteProperties(0,0)
setDV = meliso_obj.getDeviceVariation(0,0)

print(setCP)
print(setWP)
print(setDV)