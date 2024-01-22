import meliso
import numpy as np

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
meliso_obj = meliso.MelisoPy(1,32,32)

#obtain an A matrix with values between 0,1
#I have observed that having matrix between 0,1 gives the best results
scaled_A = np.random.randint(0,1000,size=(32,32))/1000.0

print(scaled_A)

#initialize weights to 0 on the memristor device
meliso_obj.initializeWeights()

#set weights on the memristor
meliso_obj.setWeights(scaled_A)

TOL = 1e-6
j=0
MAX_ITR=1
while j<MAX_ITR:
    '''
    1. we send a max signals of all 1s to see the maximum output from matrix encoded in memristor 
    2. we send min signals of all 0s to see the minimum out put from matrix encoded in memristor
    3. we send the actual x value (randomly chosen for this tutorial b/w(0,1)) 
    4. we scale the output and map it back to the range of values obtained from real matvec in software only
    '''

    #max signal to find out what the maximum value of result vector could be
    x = np.ones(32)
    #load input vector to memristor
    meliso_obj.loadInput(x)
    #perform matrix vector on hardware
    meliso_obj.matVec()
    #get results
    y_max = meliso_obj.getResults()
    #for comparison conduct the matvec on the algorithm layer as well
    real_y_max = np.dot(scaled_A,x)


    #min signal to determine what minimum value of result vector could be
    x = TOL*np.ones(32)
    # load input vector to memristor
    meliso_obj.loadInput(x)
    # perform matrix vector on hardware
    meliso_obj.matVec()
    # get results
    y_min = meliso_obj.getResults()
    # for comparison conduct the matvec on the algorithm layer as well
    real_y_min = np.dot(scaled_A,x)

    #consider the actual input vector and get hardware matvec results
    x = np.random.rand(32,1)
    meliso_obj.loadInput(x)
    meliso_obj.matVec()
    y = meliso_obj.getResults()

    #
    y_adj_max = np.zeros((32,1))
    y_adj_min = np.zeros((32,1))
    delta = np.zeros((32,1))
    sign = np.ones((32,1))
    y_result_int = np.zeros((32,1))
    y_result_int2 = np.zeros((32,1))
    real_delta = np.zeros((32,1))
    y_rescaled_mem_result = np.zeros((32,1))
    for i in range(32):
        #first adjust the outcomes so that max is always less than min
        if y_max[i]<y_min[i]:
            sign[i] =-1
            y_adj_min[i] = sign[i]*y_min[i]
            y_adj_max[i] = sign[i]*y_max[i]
        else:
            y_adj_min[i] = y_min[i]
            y_adj_max[i] = y_max[i]
        delta[i] = y_adj_max[i] - y_adj_min[i]
        if delta[i] <1e-6:
            print(sign[i],y[i],y_adj_max[i],y_adj_min[i])

        #transform the output from hardware to b/w 0,1
        y_result_int[i] = (sign[i]*y[i] - y_adj_min[i])#/delta[i]
        y_result_int2[i] = (y_result_int[i]-y_adj_min[i])/delta[i]

        real_delta[i] = real_y_max[i] - real_y_min[i]

        #rescale the hardware rescaled output (i.e. y_result_int2 (0,1)) back to the actual space
        #THIS y_rescaled_mem_result is the final outcome after rescaling
        y_rescaled_mem_result[i]=real_delta[i]*y_result_int2[i] + real_y_min[i]

    #Memristor outcome must be equivalent to real_Ax
    real_Ax = np.dot(scaled_A,x)

    print("y:", y.reshape((1, 32)))
    print("y_max:",y_max.reshape((1,32)))
    print("y_min:", y_min.reshape((1, 32)))
    print("real_delta:",real_delta.reshape((1,32)))
    print("y_result_int:", y_result_int2.reshape((1, 32)))
    print("y_rescaled:",y_rescaled_mem_result.reshape((1,32)))
    print("real_Ax:",real_Ax.reshape((1,32)))
    print(y_rescaled_mem_result.reshape((1,32))-real_Ax.reshape((1,32)))
    j = j+1


