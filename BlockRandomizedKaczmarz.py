import meliso
import numpy as np



'''
WARNING: This code does not work! It needs MPI to simulate.
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

parts = 2
m=128
n=128

p_size = int(m/parts)


MAX_ITR = 1

#obtain an A matrix with values between 0,1
#I have observed that having matrix between 0,1 gives the best results
scaled_A = np.random.randint(0,1000,size=(m,n))/1000.0
scaled_A_row_norm = np.zeros((m,1))
for i in range(m):
    scaled_A_row_norm[i] = np.linalg.norm(scaled_A[i][:])
    scaled_A[i][:] = scaled_A[i][:]/scaled_A_row_norm[i]

print(scaled_A)

w_bars = float(1.0/p_size)*np.power(scaled_A_row_norm,2)

x = np.random.rand(n,1)

b = np.dot(scaled_A,x)

meliso_obj_list = []
for p in range(parts):
    meliso_obj = meliso.MelisoPy(1,p_size,n,1e-6)

    #initialize weights to 0 on the memristor device
    meliso_obj.initializeWeights()

    start = p_size*(p)
    end = start + p_size

    #set weights on the memristor
    meliso_obj.setWeights(scaled_A[start:end][:])

    meliso_obj_list.append(meliso_obj)

    print("Initialized first part")

x = np.zeros((n,1))
k=0
while k<MAX_ITR:

    #choose a partition at random
    part_k = np.random.randint(0,parts)

    start = p_size*(part_k)
    end = start + p_size

    x_s = x
    b_s = b[start:end][:]
    #consider the actual input vector and get hardware matvec results

    meliso_obj_list[part_k].loadInput(x_s)
    meliso_obj_list[part_k].matVec()
    y = meliso_obj_list[part_k].getResults()

    weighted_projection_sum = np.zeros((n,1))
    weighted_res_sum = np.zeros((p_size,1))
    for i in range(p_size):
        w_bars_i = w_bars[start + i].reshape((1,1))
        a_i = scaled_A[i].reshape((n,1))
        weighted_projection_sum = weighted_projection_sum + w_bars_i * (y[i] - b_s[i])*a_i
        weighted_res_sum[i] = w_bars_i * (y[i] - b_s[i])

    alpha_num = np.sum(weighted_res_sum)
    alpha_denom = np.power(np.linalg.norm(weighted_projection_sum),2)

    alpha = alpha_num/alpha_denom
    x_s = x_s +alpha*weighted_projection_sum

    #real_Ax = np.dot(scaled_A,x)


    #print("y_rescaled:",y_rescaled_mem_result.reshape((1,32)))
    #print("real_Ax:",real_Ax.reshape((1,32)))
    #print(y_rescaled_mem_result.reshape((1,32))-real_Ax.reshape((1,32)))
    k = k+1


