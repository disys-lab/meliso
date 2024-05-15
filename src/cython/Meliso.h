#ifndef MELISO_H_
#define MELISO_H_

#include <cstdio>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <random>
#include <vector>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#include <math.h>
#include <iostream>
#include<map>
#include <vector>
#include <cmath>
#include <omp.h>
#include "Cell.h"
#include "Array.h"
#include "formula.h"
#include "NeuroSim.h"
#include "Param.h"
#include "IO.h"
#include "Train.h"
#include "Mapping.h"
#include "Definition.h"
#include "omp.h"
#include <ctime>

/*


MCAStatistics Structure
0: Total SubArray (synaptic core) area (m2)
1: Total Neuron (neuron peripheries) area (m2)
2: Leakage power of subArrayIH (W)
3: Leakage power of NeuronIH (W)
4: Write latency (s)
5: Write energy (J)
6: Read latency (s)
7: Read energy (J)


*/

//using namespace meliso;
//using namespace std;

#define MCA_STAT_PROPERTIES 8
#define CONDUCTANCE_PROPERTIES 6
#define WRITE_PROPERTIES 6
#define DEVICE_VARIATION_PROPERTIES 4

namespace meliso{

class Meliso {

public:
    double totalSubArrayArea, totalNeuronAreaIH,heightNeuronIH, widthNeuronIH, leakageNeuronIH;
    int rows, columns;

    double *real_A_matrix;
    double *y; //Ax = y, y must be close to b
    double *delta; //rhs of Ax=b
    double *y_adj_min; //residual b-Ax
    double* actualWeights;

    double *f;
    double *d;
    double *t;

    double *real_delta; //rhs of Ax=b
    double *real_y_adj_min; //residual b-Ax

    int p;

    double *conductanceProperties,*writeProperties,*deviceVariation;

    double *mcaStats;

    int *sign; //sign

    double TOL;
    double MAX_TOL;

    bool scalingAdjusted;
    bool considerScaling;
    bool simpleScaling;

    Meliso();
    Meliso(int,int,int,double,double,int,int);

    void initializeParam(int,int);

    void setHardwareOn(int);
    void setScalingOn(int);
    void setInterpolants(int);

    void loadInput(double *);
    void initializeWeights();
    void setWeights(double *);
    void setWeightsIncremental(double *,double);
    void getWeights();
    void matVec();
    void getResults();

    void adjustScalingLimits();
    void adjustNewtonDDScaling();
    double computeInterpolants(int, int, int, double *, double *, double *,int,int);
    double evaluatePolynomial(int,double, double *, double *,int);
    void getScalingLimits(double*, double*, double);

    void setConductanceProperties(double,double,double,double,double,double);
    void getConductanceProperties(int,int);

    void setWriteProperties(double,double,double,double,int,int);
    void getWriteProperties(int,int);

    void setDeviceVariation(double,double,double,double);
    void getDeviceVariation(int,int);

    //~Meliso();

};
}
#endif /* MODELBUILDER_H_ */
