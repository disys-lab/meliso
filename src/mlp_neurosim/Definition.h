/*******************************************************************************
* Copyright (c) 2015-2017
* School of Electrical, Computer and Energy Engineering, Arizona State University
* PI: Prof. Shimeng Yu
* All rights reserved.
*   
* This source code is part of NeuroSim - a device-circuit-algorithm framework to benchmark 
* neuro-inspired architectures with synaptic devices(e.g., SRAM and emerging non-volatile memory). 
* Copyright of the model is maintained by the developers, and the model is distributed under 
* the terms of the Creative Commons Attribution-NonCommercial 4.0 International Public License 
* http://creativecommons.org/licenses/by-nc/4.0/legalcode.
* The source code is free and you can redistribute and/or modify it
* by providing that the following conditions are met:
*   
*  1) Redistributions of source code must retain the above copyright notice,
*     this list of conditions and the following disclaimer. 
*   
*  2) Redistributions in binary form must reproduce the above copyright notice,
*     this list of conditions and the following disclaimer in the documentation
*     and/or other materials provided with the distribution.
*   
* THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
* ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
* WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
* DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
* FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
* DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
* SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
* CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
* OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
* OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
* 
* Developer list: 
*   Pai-Yu Chen     Email: pchen72 at asu dot edu 
*                     
*   Xiaochen Peng   Email: xpeng15 at asu dot edu
********************************************************************************/

// This file cannot be compiled alone. Only include this file in main.cpp.

/* Global variables */
inline Param *param = new Param(); // Parameter set

/* Inputs of training set */
std::vector< std::vector<double> >
inline Input(param->numMnistTrainImages, std::vector<double>(param->nInput));

/* Outputs of training set */
std::vector< std::vector<double> >
inline Output(param->numMnistTrainImages, std::vector<double>(param->nHide));

/* Weights from input to hidden layer */
std::vector< std::vector<double> >
inline weight1(param->nHide, std::vector<double>(param->nInput));
/* Weights from hidden layer to output layer */
std::vector< std::vector<double> >
inline weight2(param->nOutput, std::vector<double>(param->nHide));

/* Weight change of weight1 */
std::vector< std::vector<double> >
inline deltaWeight1(param->nHide, std::vector<double>(param->nInput));

/* Weight change of weight2 */
std::vector< std::vector<double> >
inline deltaWeight2(param->nOutput, std::vector<double>(param->nHide));

/*the variables to track the ΔW*/
std::vector< std::vector<double> >
inline totalDeltaWeight1(param->nHide, std::vector<double>(param->nInput));
std::vector< std::vector<double> >
inline totalDeltaWeight1_abs(param->nHide, std::vector<double>(param->nInput));
/*the variables to track the ΔW*/
std::vector< std::vector<double> >
inline totalDeltaWeight2(param->nOutput, std::vector<double>(param->nHide));
std::vector< std::vector<double> >
inline totalDeltaWeight2_abs(param->nOutput, std::vector<double>(param->nHide));

/* Inputs of testing set */
std::vector< std::vector<double> >
inline testInput(param->numMnistTestImages, std::vector<double>(param->nInput));
/* Outputs of testing set */
std::vector< std::vector<double> >
inline testOutput(param->numMnistTestImages, std::vector<double>(param->nOutput));

/* Digitized inputs of training set (an integer between 0 to 2^numBitInput-1) */
std::vector< std::vector<int> >
inline dInput(param->numMnistTrainImages, std::vector<int>(param->nInput));
/* Digitized inputs of testing set (an integer between 0 to 2^numBitInput-1) */
std::vector< std::vector<int> >
inline dTestInput(param->numMnistTestImages, std::vector<int>(param->nInput));



// the arrays for optimization
std::vector< std::vector<double> >
inline gradSquarePrev1(param->nHide, std::vector<double>(param->nInput));
std::vector< std::vector<double> >
inline gradSquarePrev2(param->nOutput, std::vector<double>(param->nHide));
std::vector< std::vector<double> >
inline gradSum1(param->nHide, std::vector<double>(param->nInput));
std::vector< std::vector<double> >
inline gradSum2(param->nOutput, std::vector<double>(param->nHide));
std::vector< std::vector<double> >
inline momentumPrev1(param->nHide, std::vector<double>(param->nInput));
std::vector< std::vector<double> >
inline momentumPrev2(param->nOutput, std::vector<double>(param->nHide));


/* # of correct prediction */
inline int correct = 0;

/* Synaptic array between input and hidden layer */
inline Array *arrayIH = new Array(param->nHide, param->nInput, param->arrayWireWidth);
/* Synaptic array between hidden and output layer */
inline Array *arrayHO = new Array(param->nOutput, param->nHide, param->arrayWireWidth);

/* Random number generator engine */
inline std::mt19937 gen;

/* NeuroSim */
inline SubArray *subArrayIH;   // NeuroSim synaptic core for arrayIH
inline SubArray *subArrayHO;   // NeuroSim synaptic core for arrayHO
/* Global properties of subArrayIH */
inline InputParameter inputParameterIH;
inline Technology techIH;
inline MemCell cellIH;
/* Global properties of subArrayHO */
inline InputParameter inputParameterHO;
inline Technology techHO;
inline MemCell cellHO;
/* Neuron peripheries below subArrayIH */
inline Adder adderIH(inputParameterIH, techIH, cellIH);
inline Mux muxIH(inputParameterIH, techIH, cellIH);
inline RowDecoder muxDecoderIH(inputParameterIH, techIH, cellIH);
inline DFF dffIH(inputParameterIH, techIH, cellIH);
inline Subtractor subtractorIH(inputParameterIH, techIH, cellIH);
/* Neuron peripheries below subArrayHO */
inline Adder adderHO(inputParameterHO, techHO, cellHO);
inline Mux muxHO(inputParameterHO, techHO, cellHO);
inline RowDecoder muxDecoderHO(inputParameterHO, techHO, cellHO);
inline DFF dffHO(inputParameterHO, techHO, cellHO);
inline Subtractor subtractorHO(inputParameterHO, techHO, cellHO);
