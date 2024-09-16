#include "Meliso.h"

namespace meliso{
//using namespace meliso;

Meliso::Meliso(){
}

void Meliso::loadInput(double *x){
        for (int k = 0; k < param->nInput; k++) {
            Input[0][k] = truncate(x[k], param->numInputLevel - 1, param->BWthreshold);
            dInput[0][k] = round(Input[0][k] * (param->numInputLevel - 1));
            //printf("x[%d],%.4f\n",k,Input[0][k]);
        }
}

void Meliso::setWeightsIncremental(double *A_matrix,double precision){

    getWeights();
    double d = 0;
    for (int k = 0; k < param->nHide; k++) {
                for (int j = 0; j < param->nInput; j++){
                    deltaWeight1[k][j] = 0;
                    d = A_matrix[k*param->nInput + j]-actualWeights[k*param->nInput + j];
                    if (fabs(d)>precision){
                        deltaWeight1[k][j] = A_matrix[k*param->nInput + j]-actualWeights[k*param->nInput + j];
                    }
                    //real_A_matrix[k*param->nInput + j] = A_matrix[k*param->nInput + j];
                }
            }
    WriteWeights();

}

//void Meliso::setWeightsIterative(double *A_matrix, double precision){
//
//
//
//}

void Meliso::setWeights(double *A_matrix){
    for (int k = 0; k < param->nHide; k++) {
                for (int j = 0; j < param->nInput; j++){
                    deltaWeight1[k][j] = A_matrix[k*param->nInput + j];
                    real_A_matrix[k*param->nInput + j] = A_matrix[k*param->nInput + j];
                }
            }
    WriteWeights();
    //if(!scalingAdjusted){
    if (considerScaling && !scalingAdjusted){
        if(simpleScaling){
            adjustScalingLimits();
        }
        else{
            adjustNewtonDDScaling();
        }
    }

}

void Meliso::getWeights(){

    for (int k = 0; k < param->nHide; k++) {
                    for (int j = 0; j < param->nInput; j++){
                        actualWeights[k*param->nInput + j] = weight1[k][j];
                    }

    }
}

void Meliso::initializeWeights(){

/* Initialize weights and map weights to conductances for hardware implementation */
	WeightInitialize();

	if (param->useHardwareInTraining)
    	WeightToConductance();

	srand(0);	// Pseudorandom number seed

	scalingAdjusted = false;
	memset(y,0,rows*sizeof(double));
    memset(delta,0,rows*sizeof(double));
    memset(y_adj_min,0,rows*sizeof(double));
    memset(sign,0,rows*sizeof(int));
    memset(real_delta,0,rows*sizeof(double));
    memset(real_y_adj_min,0,rows*sizeof(double));


}

void Meliso::matVec(){
    Train(1, 1, param->optimization_type);
    if (HybridCell *temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0]))
        WeightTransfer();
    else if(_2T1F *temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0]))
        WeightTransfer_2T1F();
    /* Here the performance metrics of subArray also includes that of neuron peripheries (see Train.cpp and Test.cpp) */
//    printf("INFO: Meliso:matVec: Read latency=%.4e s\n", subArrayIH->readLatency);// + subArrayHO->readLatency);
//    printf("INFO: Meliso:matVec: Write latency=%.4e s\n", subArrayIH->writeLatency);// + subArrayHO->writeLatency);
//    printf("INFO: Meliso:matVec: Read energy=%.4e J\n", arrayIH->readEnergy + subArrayIH->readDynamicEnergy); // + arrayHO->readEnergy + subArrayHO->readDynamicEnergy);
//    printf("INFO: Meliso:matVec: Write energy=%.4e J\n", arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy); // + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy);
    if(HybridCell* temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0])){
        printf("INFO: Meliso:matVec: Transfer latency=%.4e s\n", subArrayIH->transferLatency);// + subArrayHO->transferLatency);
        printf("INFO: Meliso:matVec: Transfer latency=%.4e s\n", subArrayIH->transferLatency);
        printf("INFO: Meliso:matVec: Transfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy);// + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
    }
    else if(_2T1F* temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0])){
        printf("INFO: Meliso:matVec: Transfer latency=%.4e s\n", subArrayIH->transferLatency);
        printf("INFO: Meliso:matVec: Transfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy);// + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
     }
	printf("\n");


	//acquire the values that can change with compute operations
    mcaStats[4] = mcaStats[4] + subArrayIH->writeLatency;
    mcaStats[5] = mcaStats[5] + arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy;
    mcaStats[6] = mcaStats[6] + subArrayIH->readLatency;
    mcaStats[7] = mcaStats[7] + arrayIH->readEnergy + subArrayIH->readDynamicEnergy;

    printf("INFO: Meliso:matVec: Read latency=%.4e s\n", mcaStats[6]);// + subArrayHO->readLatency);
    printf("INFO: Meliso:matVec: Write latency=%.4e s\n", mcaStats[4]);// + subArrayHO->writeLatency);
    printf("INFO: Meliso:matVec: Read energy=%.4e J\n", mcaStats[7]); // + arrayHO->readEnergy + subArrayHO->readDynamicEnergy);
    printf("INFO: Meliso:matVec: Write energy=%.4e J\n", mcaStats[5]); // + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy);
}

void Meliso::getResults(){
      //memset(y,0,rows*sizeof(double));
      for (int k = 0; k < param->nHide; k++) {
            y[k] = Output[0][k];
//            //if (scalingAdjusted){
//            if ( considerScaling && scalingAdjusted){
//                if(simpleScaling){
//                    y[k] = (sign[k]*Output[0][k] - y_adj_min[k])/delta[k];
//                    y[k] = real_delta[k]*y[k] + real_y_adj_min[k];
//                    //printf("getResults:: real_delta[%d]=%f\n",k,real_delta[k]);
//                    //printf("getResults:: real_delta[%d]=%f\n",k,real_delta[k]);
//                    //printf("getResults:: delta[%d]=%f\n",k,delta[k]);
//                    //printf("getResults:: scaling adjusted : %d consider scaling %d\n",scalingAdjusted,sc);
//                }
//                else{
//                    y[k] = Output[0][k] - evaluatePolynomial(p-1,Output[0][k],f,t,k*p);
//                    //printf("INFO: Meliso:getResults :%d y=%lf Output=%lf\n",k,y[k],Output[0][k]);
//                }
//            }
//            else{
//                y[k] = Output[0][k];
//                //printf("INFO: Meliso:getResults turnOnScaling=%d::scalingAdjusted=%d :: Output[%d] = %lf\n",considerScaling,scalingAdjusted,k,y[k]);
//            }
        }
}

void Meliso::getScalingLimits(double *y_scaled_result,double *real_y_result, double input_value){
    double *x = (double*) malloc(columns*sizeof(double));
    memset(x,0,columns*sizeof(double));

    for (int i=0;i<columns;i++){
        x[i] = input_value;
    }

    loadInput(x);

    matVec();

    for (int i=0;i<rows;i++){
        y_scaled_result[i] = Output[0][i];
    }

    for (int k = 0; k < param->nHide; k++) {
        for (int j = 0; j < param->nInput; j++){
            //real_y_result[k] = real_y_result[k] + x[j]*x[j];
            real_y_result[k] = real_y_result[k] + real_A_matrix[k*param->nInput + j]*x[j];
        }
    }

}

void Meliso::adjustScalingLimits(){

    getScalingLimits(delta,real_delta,MAX_TOL);
    getScalingLimits(y_adj_min,real_y_adj_min,TOL);

    for (int i=0; i< rows;i++){
        sign[i] = 1;
        if(delta[i] < y_adj_min[i]){
            sign[i] = -1;
            delta[i] = sign[i]*delta[i];
            y_adj_min[i] = sign[i]*y_adj_min[i];
        }
        delta[i] = delta[i] - y_adj_min[i];
        real_delta[i] = real_delta[i] - real_y_adj_min[i];
    }

    scalingAdjusted = true;
}

void Meliso::adjustNewtonDDScaling(){

    int m = rows;
    printf("INFO: Meliso:adjustNewtonDDScaling: adjustNewtonDDScaling interpolants=%d\n",p);
    for (int a = 0; a<p; a++){

        double value;// = a*double(MAX_TOL)/(p-1);

        initializeWeights();

        for (int k = 0; k < param->nHide; k++) {
                    for (int j = 0; j < param->nInput; j++){
                        value = ((double)rand())/(RAND_MAX)*a/(p-1);
                        deltaWeight1[k][j] = value;
                        real_A_matrix[k*param->nInput + j] = value;
                    }
                }

        WriteWeights();

        getScalingLimits(delta,real_delta,MAX_TOL);
        for (int i=0 ;i <m; i++){
            t[i*p+a] = delta[i];
            d[i*p+a] = delta[i] - real_delta[i];
            printf("INFO: Meliso:adjustNewtonDDScaling :value=%lf t[%d]=%lf, d[%d]=%lf\n",value,i*p+a,i*p+a,t[i*p+a],d[i*p+a]);
        }

        memset(delta,0,m*sizeof(double));
        memset(real_delta,0,m*sizeof(double));
    }

    for(int i= 0; i<m; i++){
        computeInterpolants(p-1,0,p-1,f,d,t,1,i*p);
    }

    scalingAdjusted = true;
}


double Meliso::computeInterpolants(int n, int start, int end, double *f, double *d, double *t,int store_res,int arr_start){

    if(start == end){
        return d[arr_start+start];
    }

    double f_2k = computeInterpolants(n-1,start+1,end,f,d,t,0,arr_start);
    double f_1km1 = computeInterpolants(n-1,start,end-1,f,d,t,1,arr_start);

    double f_res = double(f_2k - f_1km1)/(t[arr_start+end] - t[arr_start+start]);

    if(store_res){
        f[arr_start+n] = f_res;
        f[arr_start+n-1] = f_1km1;
    }

    return f_res;

}

double Meliso::evaluatePolynomial(int n,double value, double *f, double *t, int arr_start){

    double res = 0;

    for (int i=0;i<n;i++){
        double prod = 1;
        for (int j =0; j<i ; j++){
            prod = prod *(value -t[arr_start+j]);
        }

        res = res + prod*f[arr_start+i];
    }

    return res;

}

void Meliso::setConductanceProperties(  double maxConductance,
                                        double minConductance,
                                        double avgMaxConductance,
                                        double avgMinConductance,
                                        double conductance,
                                        double conductancePrev
                                     ){
        //ConductanceProperties:
        //    maxConductance: 3.846153846e-8 # To be dynamically determined or specified
        //    minConductance: 3.076923077e-9 # To be dynamically determined or specified
        //    avgMaxConductance: maxConductance # To be dynamically determined or specified
        //    avgMinConductance: minConductance # To be dynamically determined or specified
        //    conductance: minConductance # This will be dynamically updated
        //    conductancePrev: conductance # This will be dynamically updated
        for (int k = 0; k < param->nInput; k++) {
            for (int j = 0; j < param->nHide; j++) {
                //only if the Cell can be cast as an AnalogNVM
                if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[j][k])) {	// Analog eNVM
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->maxConductance = maxConductance;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->minConductance = minConductance;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->avgMaxConductance = avgMaxConductance;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->avgMinConductance = avgMinConductance;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance = conductance;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductancePrev = conductancePrev;
            }
          }
        }
}

void Meliso::getConductanceProperties(int j,int k){
    memset(conductanceProperties,0,6*sizeof(double));
    if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[j][k])) {	// Analog eNVM
            conductanceProperties[0] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->maxConductance;
            conductanceProperties[1] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->minConductance;
            conductanceProperties[2] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->avgMaxConductance;
            conductanceProperties[3] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->avgMinConductance;
            conductanceProperties[4] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductance;
            conductanceProperties[5] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->conductancePrev;
    }


}

void Meliso::setWriteProperties(  double writeVoltageLTP,
                                        double writeVoltageLTD,
                                        double writePulseWidthLTP,
                                        double writePulseWidthLTD,
                                        int maxNumLevelLTP,
                                        int maxNumLevelLTD
                                     ){
        //WriteProperties:
        //    writeVoltageLTP: 3.20 # To be specified based on device type
        //    writeVoltageLTD: -2.8 # To be specified based on device type
        //    writePulseWidthLTP: 300e-6 # To be specified based on device type
        //    writePulseWidthLTD: 300e-6 # To be specified based on device type
        //    maxNumLevelLTP: [2, 4, 8, 16, 32, 64, 97, 128, 256, 512, 1024] # To be specified based on device type
        //    maxNumLevelLTD: [2, 4, 8, 16, 32, 64, 100, 128, 256, 512, 1024] # To be specified based on device type

        for (int k = 0; k < param->nInput; k++) {
            for (int j = 0; j < param->nHide; j++) {
                //only if the Cell can be cast as an AnalogNVM
                if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[j][k])) {	// Analog eNVM
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeVoltageLTP = writeVoltageLTP;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeVoltageLTD = writeVoltageLTD;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writePulseWidthLTP = writePulseWidthLTP;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writePulseWidthLTD = writePulseWidthLTD;
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->maxNumLevelLTP = double(maxNumLevelLTP);
                        static_cast<AnalogNVM*>(arrayIH->cell[j][k])->maxNumLevelLTD = double(maxNumLevelLTD);
            }
          }
        }
}

void Meliso::getWriteProperties(int j,int k){
    memset(writeProperties,0,6*sizeof(double));
    if (AnalogNVM *temp = dynamic_cast<AnalogNVM*>(arrayIH->cell[j][k])) {	// Analog eNVM
            writeProperties[0] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeVoltageLTP;
            writeProperties[1] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writeVoltageLTD;
            writeProperties[2] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writePulseWidthLTP;
            writeProperties[3] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->writePulseWidthLTD;
            writeProperties[4] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->maxNumLevelLTP;
            writeProperties[5] = static_cast<AnalogNVM*>(arrayIH->cell[j][k])->maxNumLevelLTD;
    }


}

void Meliso::setDeviceVariation(    double NL_LTP,
                                    double NL_LTD,
                                    double sigmaDtoDvar,
                                    double sigmaCtoCvar
                               ){
        //DeviceVariation:
        //    NL_LTP: 2.4 # Specify if nonlinear write is enabled
        //    NL_LTD: -4.88 # Specify if nonlinear write is enabled
        //    sigmaDtoD: 0 # Sigma for device-to-device variation, specify if applicable
        //    sigmaCtoC: 0.035 #* (maxConductance - minConductance) # Sigma for cycle-to-cycle variation, specify if applicable, may depend on memory window

        for (int k = 0; k < param->nInput; k++) {
            for (int j = 0; j < param->nHide; j++) {
                //only if the Cell can be cast as an AnalogNVM
                if (RealDevice *temp = dynamic_cast<RealDevice*>(arrayIH->cell[j][k])) {	// Analog eNVM
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->NL_LTP = NL_LTP;
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->NL_LTD = NL_LTD;
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->sigmaDtoD = sigmaDtoDvar;
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->sigmaCtoC = sigmaCtoCvar;

                        double maxNumLevelLTP = static_cast<RealDevice*>(arrayIH->cell[j][k])->maxNumLevelLTP;
                        double maxNumLevelLTD = static_cast<RealDevice*>(arrayIH->cell[j][k])->maxNumLevelLTD;

                        double maxConductance = static_cast<RealDevice*>(arrayIH->cell[j][k])->maxConductance;
                        double minConductance = static_cast<RealDevice*>(arrayIH->cell[j][k])->minConductance;

                        std::mt19937 localGen;	// It's OK not to use the external gen, since here the device-to-device vairation is a one-time deal
	                    localGen.seed(std::time(0));

                        static_cast<RealDevice*>(arrayIH->cell[j][k])->gaussian_dist2 = new std::normal_distribution<double>(0, sigmaDtoDvar);	// Set up mean and stddev for device-to-device weight update vairation
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->paramALTP = getParamA(NL_LTP + (*static_cast<AnalogNVM*>(arrayIH->cell[j][k])->gaussian_dist2)(localGen)) * maxNumLevelLTP;	// Parameter A for LTP nonlinearity
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->paramALTD = getParamA(NL_LTD + (*static_cast<AnalogNVM*>(arrayIH->cell[j][k])->gaussian_dist2)(localGen)) * maxNumLevelLTD;	// Parameter A for LTD nonlinearity

                        /* Cycle-to-cycle weight update variation */
                        //static_cast<AnalogNVM*>(arrayIH->cell[j][k])->sigmaCtoC = sigmaC2Cvar* (maxConductance - minConductance);	// Sigma of cycle-to-cycle weight update vairation: defined as the percentage of conductance range
                        static_cast<RealDevice*>(arrayIH->cell[j][k])->gaussian_dist3 = new std::normal_distribution<double>(0, sigmaCtoCvar* (maxConductance - minConductance));    // Set up mean and stddev for cycle-to-cycle weight update vairation
            }
          }
        }
}

void Meliso::getDeviceVariation(int j,int k){
    memset(deviceVariation,0,4*sizeof(double));
    if (AnalogNVM *temp = dynamic_cast<RealDevice*>(arrayIH->cell[j][k])) {	// Analog eNVM
            deviceVariation[0] = static_cast<RealDevice*>(arrayIH->cell[j][k])->NL_LTP;
            deviceVariation[1] = static_cast<RealDevice*>(arrayIH->cell[j][k])->NL_LTD;
            deviceVariation[2] = static_cast<RealDevice*>(arrayIH->cell[j][k])->sigmaDtoD;
            deviceVariation[3] = static_cast<RealDevice*>(arrayIH->cell[j][k])->sigmaCtoC;
    }
}

void Meliso::initializeParam(int m,int n){

    param = new Param(); // Parameter set
    param->nInput = n;
    param->nHide = m;
    param->numMnistTrainImages = 1;
    param->numMnistTestImages = 1;

    weight1 = std::vector< std::vector<double> > (param->nHide, std::vector<double>(param->nInput));
//    weight2 = std::vector< std::vector<double> > (param->nOutput, std::vector<double>(param->nHide));
    real_weight1 = std::vector< std::vector<double> > (param->nHide, std::vector<double>(param->nInput));
    deltaWeight1 = std::vector< std::vector<double> > (param->nHide, std::vector<double>(param->nInput));
    totalDeltaWeight1 = std::vector< std::vector<double> > (param->nHide, std::vector<double>(param->nInput));
    totalDeltaWeight1_abs = std::vector< std::vector<double> > (param->nHide, std::vector<double>(param->nInput));

    arrayIH = new Array(param->nHide, param->nInput, param->arrayWireWidth);
    Input = std::vector< std::vector<double> > (param->numMnistTrainImages, std::vector<double>(param->nInput));
    Output = std::vector< std::vector<double> > (param->numMnistTrainImages, std::vector<double>(param->nHide));

    dInput = std::vector< std::vector<int> > (param->numMnistTrainImages, std::vector<int>(param->nInput));

}

void Meliso::setHardwareOn(int turnOnHardware){

    if(turnOnHardware){
        param->useHardwareInTrainingFF = true;
        param->useHardwareInTrainingWU = true;
    }
    else{
        param->useHardwareInTrainingFF = false;
        param->useHardwareInTrainingWU = false;
    }

}

void Meliso::setScalingOn(int turnOnScaling){

    if (turnOnScaling){
	    considerScaling = true;
    }
    else{
        considerScaling = false;
        //printf("setScaling:: scaling adjusted : %d consider scaling %d\n",scalingAdjusted,considerScaling);
    }
    //
}

void Meliso::setInterpolants(int interpolants){

    p = interpolants;

    int m = rows;
    f = (double*)malloc(p*m*sizeof(double));
    memset(f,0,m*sizeof(double));

    d = (double*)malloc(p*m*sizeof(double));
    memset(y,0,m*sizeof(double));

    t = (double*)malloc(p*m*sizeof(double));
    memset(t,0,m*sizeof(double));

    if(!simpleScaling){
        adjustNewtonDDScaling();
    }
    printf("INFO: Meliso:Constructor: scalingAdjusted=%d \n",scalingAdjusted);
}

Meliso::Meliso(int device_type,int m,int n, double max_tol,double min_tol,int turnOnHardware,int turnOnScaling) {

    p = 3;

    rows = m;
    columns = n;

    TOL = min_tol;
    MAX_TOL = max_tol;

    initializeParam(m,n);

	gen.seed(0);

	scalingAdjusted = false;
	considerScaling = false;
	simpleScaling = false;

    setHardwareOn(turnOnHardware);
    setScalingOn(turnOnScaling);

    mcaStats = (double*)malloc(MCA_STAT_PROPERTIES*sizeof(double));
    memset(mcaStats,0,MCA_STAT_PROPERTIES*sizeof(double));

    conductanceProperties = (double*)malloc(CONDUCTANCE_PROPERTIES*sizeof(double));
    memset(conductanceProperties,0,CONDUCTANCE_PROPERTIES*sizeof(double));

    writeProperties = (double*)malloc(WRITE_PROPERTIES*sizeof(double));
    memset(writeProperties,0,WRITE_PROPERTIES*sizeof(double));

    deviceVariation = (double*)malloc(DEVICE_VARIATION_PROPERTIES*sizeof(double));
    memset(deviceVariation,0,DEVICE_VARIATION_PROPERTIES*sizeof(double));

    real_A_matrix = (double*)malloc(m*n*sizeof(double));
    memset(real_A_matrix,0,m*n*sizeof(double));

    actualWeights = (double*)malloc(m*n*sizeof(double));
    memset(actualWeights,0,m*n*sizeof(double));

	y = (double*)malloc(m*sizeof(double));
    memset(y,0,m*sizeof(double));

    delta = (double*)malloc(m*sizeof(double));
    memset(delta,0,m*sizeof(double));

    y_adj_min = (double*)malloc(m*sizeof(double));
    memset(y_adj_min,0,m*sizeof(double));

    sign = (int*)malloc(m*sizeof(int));
    memset(sign,0,m*sizeof(int));

    real_delta = (double*)malloc(m*sizeof(double));
    memset(real_delta,0,m*sizeof(double));

    real_y_adj_min = (double*)malloc(m*sizeof(double));
    memset(real_y_adj_min,0,m*sizeof(double));




	/* Initialization of synaptic array from input to hidden layer */

	if (device_type == 0){
        arrayIH->Initialization<IdealDevice>();
    }
	else if (device_type == 1){
        arrayIH->Initialization<RealDevice>();
    }
    else if (device_type == 2){
        arrayIH->Initialization<MeasuredDevice>();
    }
    else if (device_type == 3){
        arrayIH->Initialization<SRAM>(param->numWeightBit);
    }
    else if (device_type == 4){
        arrayIH->Initialization<DigitalNVM>(param->numWeightBit,true);
    }
    else if (device_type == 5){
        arrayIH->Initialization<HybridCell>(); // the 3T1C+2PCM cell
    }
    else if (device_type == 6){
        arrayIH->Initialization<_2T1F>();
    }
    else{
        printf("INFO: Meliso:Constructor:Unsupported device_type\n");
        exit(0);
    }
    omp_set_num_threads(16);

	/* Initialization of NeuroSim synaptic cores */
	param->relaxArrayCellWidth = 0;
	NeuroSimSubArrayInitialize(subArrayIH, arrayIH, inputParameterIH, techIH, cellIH);
	param->relaxArrayCellWidth = 1;

	/* Calculate synaptic core area */
	NeuroSimSubArrayArea(subArrayIH);

	/* Calculate synaptic core standby leakage power */
	NeuroSimSubArrayLeakagePower(subArrayIH);

	/* Initialize the neuron peripheries */
	NeuroSimNeuronInitialize(subArrayIH, inputParameterIH, techIH, cellIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);

	NeuroSimNeuronArea(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH, &heightNeuronIH, &widthNeuronIH);
	leakageNeuronIH = NeuroSimNeuronLeakagePower(subArrayIH, adderIH, muxIH, muxDecoderIH, dffIH, subtractorIH);


	/* Print the area of synaptic core and neuron peripheries */
	totalSubArrayArea = subArrayIH->usedArea; //+ subArrayHO->usedArea;
	totalNeuronAreaIH = adderIH.area + muxIH.area + muxDecoderIH.area + dffIH.area + subtractorIH.area;

	printf("INFO: Meliso:Constructor: Total SubArray (synaptic core) area=%.4e m^2\n", totalSubArrayArea);
	printf("INFO: Meliso:Constructor: Total Neuron (neuron peripheries) area=%.4e m^2\n", totalNeuronAreaIH); // + totalNeuronAreaHO);
	printf("INFO: Meliso:Constructor: Total area=%.4e m^2\n", totalSubArrayArea + totalNeuronAreaIH); //+ totalNeuronAreaHO);

	/* Print the standby leakage power of synaptic core and neuron peripheries */
	printf("INFO: Meliso:Constructor: Leakage power of subArrayIH is : %.4e W\n", subArrayIH->leakage);

	printf("INFO: Meliso:Constructor: Leakage power of NeuronIH is : %.4e W\n", leakageNeuronIH);
	printf("INFO: Meliso:Constructor: Total leakage power of subArray is : %.4e W\n", subArrayIH->leakage);// + subArrayHO->leakage);
	printf("INFO: Meliso:Constructor: Total leakage power of Neuron is : %.4e W\n", leakageNeuronIH); // + leakageNeuronHO);

    //only initialized during construction of Meliso
    mcaStats[0] = totalSubArrayArea;
    mcaStats[1] = totalNeuronAreaIH;
    mcaStats[2] = subArrayIH->leakage;
    mcaStats[3] = leakageNeuronIH;

    //subArrayIH->writeLatency
    //subArrayIH->readLatency
    //subArrayIH->writeDynamicEnergy
    //subArrayIH->readDynamicEnergy




}
}
