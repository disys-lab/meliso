# distutils: language=c++

from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from Meliso cimport Meliso
from libc.stdlib cimport malloc, free
cimport cython
from cython.view cimport array as cvarray
from cpython.pycapsule cimport *

import numpy as np

# Create a Cython extension type which holds a C++ instance
# as an attribute and create a bunch of forwarding methods
# Python extension type.

cdef class MelisoPy:
    cdef Meliso melisoObj  # Hold a C++ instance which we're wrapping

    cdef int m
    cdef int n

    cdef double TOL
    cdef double MAX_TOL
    cdef int device_type

    cdef double *A_matrix
    cdef double *x
    cdef double *y

    def __cinit__(self,device_type,rows,columns,MAX_TOL,TOL,turnOnHardware,turnOnScaling):
        self.m = rows
        self.n = columns

        self.TOL = TOL
        self.TOL = MAX_TOL

        self.A_matrix = <double *>malloc(self.m*self.n*cython.sizeof(double))
        self.x = <double*>malloc(self.n*cython.sizeof(double))
        self.y = <double*>malloc(self.m*cython.sizeof(double))
        self.device_type = device_type

        HardwareOn = 0
        if turnOnHardware:
            HardwareOn = 1

        ScalingOn = 0
        if turnOnScaling:
            ScalingOn = 1

        self.melisoObj = Meliso(self.device_type,self.m,self.n,MAX_TOL,TOL,HardwareOn,ScalingOn)

    def setHardwareOn(self,turnOnHardware):
        self.melisoObj.setHardwareOn(turnOnHardware)

    def setScalingOn(self,turnOnScaling):
        self.melisoObj.setScalingOn(turnOnScaling)

    def initializeWeights(self):
        self.melisoObj.initializeWeights()

    def setWeights(self,np_A_matrix):
        ctr = 0

        for i in xrange(self.m):
            for j in xrange(self.n):
                self.A_matrix[ctr] = np_A_matrix[i][j]
                ctr = ctr +1

        self.melisoObj.setWeights(self.A_matrix)

    def loadInput(self,np_x):
        for j in xrange(self.n):
            self.x[j] = np_x[j]
        self.melisoObj.loadInput(self.x)

    def matVec(self):
        self.melisoObj.matVec()

    def getResults(self):
        self.melisoObj.getResults()
        y = np.zeros(self.m,dtype=float).reshape((self.m,1))
        for i in xrange(self.m):
            y[i][0] = self.melisoObj.y[i]
        return y

    def setConductanceProperties(self,maxConductance,minConductance,avgMaxConductance,avgMinConductance,conductance,conductancePrev):
        self.melisoObj.setConductanceProperties(maxConductance,minConductance,avgMaxConductance,avgMinConductance,conductance,conductancePrev)

    def getConductanceProperties(self,x,y):
        self.melisoObj.getConductanceProperties(x,y)

        conductanceProperties = np.zeros(6,dtype=float).reshape((6,1))
        for i in xrange(6):
            conductanceProperties[i][0] = self.melisoObj.conductanceProperties[i]

        return conductanceProperties

    def setWriteProperties(self,writeVoltageLTP,writeVoltageLTD,writePulseWidthLTP,writePulseWidthLTD,maxNumLevelLTP,maxNumLevelLTD):
        self.melisoObj.setWriteProperties(writeVoltageLTP,writeVoltageLTD,writePulseWidthLTP,writePulseWidthLTD,maxNumLevelLTP,maxNumLevelLTD)

    def getWriteProperties(self,x,y):
        self.melisoObj.getWriteProperties(x,y)

        writeProperties = np.zeros(6,dtype=float).reshape((6,1))
        for i in xrange(6):
            writeProperties[i][0] = self.melisoObj.writeProperties[i]

        return writeProperties

    def setConductanceProperties(self,maxConductance,minConductance,avgMaxConductance,avgMinConductance,conductance,conductancePrev):
        self.melisoObj.setConductanceProperties(maxConductance,minConductance,avgMaxConductance,avgMinConductance,conductance,conductancePrev)

    def getConductanceProperties(self,x,y):
        self.melisoObj.getConductanceProperties(x,y)

        conductanceProperties = np.zeros(6,dtype=float).reshape((6,1))
        for i in xrange(6):
            conductanceProperties[i][0] = self.melisoObj.conductanceProperties[i]

        return conductanceProperties

    def setDeviceVariation(self,NL_LTP, NL_LTD,sigmaDtoD,sigmaCtoC):
        self.melisoObj.setDeviceVariation(NL_LTP, NL_LTD,sigmaDtoD,sigmaCtoC)

    def getDeviceVariation(self,x,y):
        self.melisoObj.getDeviceVariation(x,y)

        deviceVariation = np.zeros(4,dtype=float).reshape((4,1))
        for i in xrange(4):
            deviceVariation[i][0] = self.melisoObj.deviceVariation[i]

        return deviceVariation

    def getMCAStats(self,num_mca_stats):
        mcaStats = np.zeros(num_mca_stats,dtype=float).reshape((num_mca_stats,1))
        for i in range(num_mca_stats):
            mcaStats[i][0] = self.meliisoObj.mcaStats[i]

        return mcaStats