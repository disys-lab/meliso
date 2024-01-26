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


//using namespace meliso;
//using namespace std;

namespace meliso{

class Meliso {

public:
    double totalSubArrayArea, totalNeuronAreaIH,heightNeuronIH, widthNeuronIH, leakageNeuronIH;
    int rows, columns;

    double *real_A_matrix;
    double *y; //Ax = y, y must be close to b
    double *delta; //rhs of Ax=b
    double *y_adj_min; //residual b-Ax

    double *real_delta; //rhs of Ax=b
    double *real_y_adj_min; //residual b-Ax

    int *sign; //sign

    double TOL;
    double MAX_TOL;

    bool scalingAdjusted;

    Meliso();
    Meliso(int,int,int,double,double,int);

    void loadInput(double *);
    void initializeWeights();
    void setWeights(double *);
    void matVec();
    void getResults();

    void adjustScalingLimits();
    void getScalingLimits(double*, double*, double);

    //~Meliso();

};
}
#endif /* MODELBUILDER_H_ */
