from libcpp.map cimport map
from libcpp.string cimport string
from libcpp.vector cimport vector
from cython.operator cimport dereference, preincrement
from Meliso cimport Meliso

cdef extern from "Meliso.cpp":
        pass

# Declare the class with cdef
cdef extern from "Meliso.h" namespace "meliso":
    cdef cppclass Meliso:
        int device_type
        Meliso() except +
        Meliso(int,int,int,double,double,int,int) except +

        double totalSubArrayArea, totalNeuronAreaIH,heightNeuronIH, widthNeuronIH, leakageNeuronIH;

        double *y;
        double *delta;
        double *y_min;
        double *conductanceProperties;
        double *writeProperties;
        double *deviceVariation;

        double *mcaStats;

        int* sign;

        void loadInput(double *)
        void initializeWeights()
        void setWeights(double *)
        void matVec()
        void getResults()
        void setConductanceProperties(double,double,double,double,double,double)
        void getConductanceProperties(int,int)

        void setWriteProperties(double,double,double,double,double,double)
        void getWriteProperties(int,int)

        void setDeviceVariation(double,double,double,double)
        void getDeviceVariation(int,int)


