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
        Meliso(int,int,int,double) except +

        double totalSubArrayArea, totalNeuronAreaIH,heightNeuronIH, widthNeuronIH, leakageNeuronIH;

        double *y,*delta,*y_min;
        int* sign;

        void loadInput(double *)
        void initializeWeights()
        void setWeights(double *)
        void matVec()
        void getResults()
