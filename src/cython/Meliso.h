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
//    double *A_matrix;
//    double *x;
    double *y;
    Meliso();
    Meliso(int,int,int);

    void loadInput(double *);
    void initializeWeights();
    void setWeights(double *);
    void matVec();
    void getResults();

    //~Meliso();

};
}
#endif /* MODELBUILDER_H_ */
