import meliso
import numpy as np

meliso_obj = meliso.MelisoPy(1,32,32)

A = np.random.rand(32,32)



#scale between -1 and 1
scaled_A = (2.)*(A - np.min(A))/np.ptp(A)-1

print(scaled_A)

meliso_obj.initializeWeights()
meliso_obj.setWeights(scaled_A)
j=0
while j<10:

    x = np.ones(32)
    meliso_obj.loadInput(x)
    meliso_obj.matVec()
    y_max = meliso_obj.getResults()
    real_y_max = np.dot(scaled_A,x)


    x = 0*np.ones(32)
    meliso_obj.loadInput(x)
    meliso_obj.matVec()
    y_min = meliso_obj.getResults()
    real_y_min = np.dot(scaled_A,x)


    x = np.random.rand(32,1)
    meliso_obj.loadInput(x)
    meliso_obj.matVec()
    y = meliso_obj.getResults()


    y_adj_max = np.zeros((32,1))
    y_adj_min = np.zeros((32,1))
    delta = np.zeros((32,1))
    sign = np.zeros((32,1))
    y_result_int = np.zeros((32,1))
    y_result_int2 = np.zeros((32,1))
    real_delta = np.zeros((32,1))
    y_rescaled_mem_result = np.zeros((32,1))
    for i in range(32):
        if y_max[i]<y_min[i]:
            sign[i] =-1
            y_adj_min[i] = sign[i]*y_min[i]
            y_adj_max[i] = sign[i]*y_max[i]
        delta[i] = y_adj_max[i] - y_adj_min[i]
        if delta[i] <1e-6:
            print(y_adj_max[i],y_adj_min[i])

        y_result_int[i] = (sign[i]*y[i] - y_adj_min[i])#/delta[i]

        y_result_int2[i] = (y_result_int[i]-y_adj_min[i])/delta[i]
        real_delta[i] = real_y_max[i] - real_y_min[i]
        y_rescaled_mem_result[i]=real_delta[i]*y_result_int2[i] + real_y_min[i]


    real_Ax = np.dot(scaled_A,x)

    # print(y_max)
    print(y_rescaled_mem_result)
    print(real_Ax)
    j = j+1


