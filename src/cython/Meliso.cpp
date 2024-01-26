#include "Meliso.h"

namespace meliso{
//using namespace meliso;

Meliso::Meliso(){
}

void Meliso::loadInput(double *x){
        for (int k = 0; k < param->nInput; k++) {
            Input[0][k] = truncate(x[k], param->numInputLevel - 1, param->BWthreshold);
            dInput[0][k] = round(Input[0][k] * (param->numInputLevel - 1));
        }
}

void Meliso::setWeights(double *A_matrix){
    for (int k = 0; k < param->nInput; k++) {
                for (int j = 0; j < param->nHide; j++){
                    deltaWeight1[k][j] = A_matrix[k*param->nHide + j];
                    real_A_matrix[k*param->nHide + j] = A_matrix[k*param->nHide + j];
                }
            }
    WriteWeights();
    if (!scalingAdjusted)
        adjustScalingLimits();

}

void Meliso::initializeWeights(){

/* Initialize weights and map weights to conductances for hardware implementation */
	WeightInitialize();

	if (param->useHardwareInTraining)
    	WeightToConductance();

	srand(0);	// Pseudorandom number seed

}

void Meliso::matVec(){

    Train(1, 1, param->optimization_type);

    if (HybridCell *temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0]))
        WeightTransfer();
    else if(_2T1F *temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0]))
        WeightTransfer_2T1F();
    /* Here the performance metrics of subArray also includes that of neuron peripheries (see Train.cpp and Test.cpp) */
    printf("\tRead latency=%.4e s\n", subArrayIH->readLatency);// + subArrayHO->readLatency);
    printf("\tWrite latency=%.4e s\n", subArrayIH->writeLatency);// + subArrayHO->writeLatency);
    printf("\tRead energy=%.4e J\n", arrayIH->readEnergy + subArrayIH->readDynamicEnergy); // + arrayHO->readEnergy + subArrayHO->readDynamicEnergy);
    printf("\tWrite energy=%.4e J\n", arrayIH->writeEnergy + subArrayIH->writeDynamicEnergy); // + arrayHO->writeEnergy + subArrayHO->writeDynamicEnergy);
    if(HybridCell* temp = dynamic_cast<HybridCell*>(arrayIH->cell[0][0])){
        printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);// + subArrayHO->transferLatency);
        printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);
        printf("\tTransfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy);// + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
    }
    else if(_2T1F* temp = dynamic_cast<_2T1F*>(arrayIH->cell[0][0])){
        printf("\tTransfer latency=%.4e s\n", subArrayIH->transferLatency);
        printf("\tTransfer energy=%.4e J\n", arrayIH->transferEnergy + subArrayIH->transferDynamicEnergy);// + arrayHO->transferEnergy + subArrayHO->transferDynamicEnergy);
     }
	printf("\n");
}

void Meliso::getResults(){
      for (int k = 0; k < param->nHide; k++) {
            //y[k] = (sign[k]*Output[0][k] -y_adj_min[k])/(MAX_TOL-TOL);
            if (scalingAdjusted){
                y[k] = (sign[k]*Output[0][k] - y_adj_min[k])/delta[k];
                y[k] = real_delta[k]*y[k] + real_y_adj_min[k];
            }
            else{
                y[k] = Output[0][k];

            }
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

    for (int k = 0; k < param->nInput; k++) {
                for (int j = 0; j < param->nHide; j++){
                    real_y_result[k] = real_y_result[k] + real_A_matrix[k*param->nHide + j]*x[j];
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

Meliso::Meliso(int device_type,int m,int n, double max_tol,double min_tol,int turnOnHardware) {
    rows = m;
    columns = n;

    TOL = min_tol;
    MAX_TOL = max_tol;

    param->nInput = n;
    param->nHide = m;

	gen.seed(0);

    if(turnOnHardware){
        param->useHardwareInTrainingFF = true;
        param->useHardwareInTrainingWU = true;
    }
    else{
        param->useHardwareInTrainingFF = false;
        param->useHardwareInTrainingWU = false;
    }
	scalingAdjusted = false;

    real_A_matrix = (double*)malloc(m*n*sizeof(double));
    memset(real_A_matrix,0,m*n*sizeof(double));

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
        printf("Unsupported device_type\n");
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

	printf("Total SubArray (synaptic core) area=%.4e m^2\n", totalSubArrayArea);
	printf("Total Neuron (neuron peripheries) area=%.4e m^2\n", totalNeuronAreaIH); // + totalNeuronAreaHO);
	printf("Total area=%.4e m^2\n", totalSubArrayArea + totalNeuronAreaIH); //+ totalNeuronAreaHO);

	/* Print the standby leakage power of synaptic core and neuron peripheries */
	printf("Leakage power of subArrayIH is : %.4e W\n", subArrayIH->leakage);

	printf("Leakage power of NeuronIH is : %.4e W\n", leakageNeuronIH);
	printf("Total leakage power of subArray is : %.4e W\n", subArrayIH->leakage);// + subArrayHO->leakage);
	printf("Total leakage power of Neuron is : %.4e W\n", leakageNeuronIH); // + leakageNeuronHO);



}
}
