import meliso
import numpy as np
from mpi4py import MPI

import matplotlib.pyplot as plt

def parallelMatVec(x_val):
    x = np.copy(x_val)
    # send chunks of x to all processes on chosen row
    for col_id in range(col_parts):
        start = col_id * col_p_size
        end = (col_id + 1) * col_p_size
        dest_rank = row_id * col_parts + col_id
        comm.Send(x[start:end], dest=dest_rank)

    sum_y = np.zeros(row_p_size, dtype=np.float64)

    for col_id in range(col_parts):
        source_rank = row_id * col_parts + col_id
        y = np.empty(row_p_size, dtype=np.float64)
        comm.Recv(y, source=source_rank)
        sum_y = sum_y + y


    return sum_y

def globalRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,row_size,i,b_norm):
    '''
    This is the classical Randomized Kaczmarz. The problem here is in the scaling of x_updated
    Notes 1/24/24: I think this is primarily the reason why the other two were not converging.
    We need to ensure that 0<= x <=1. This is the critical bottleneck for any Memristor based solver to succeed
    '''

    a_i = scaled_A[i].reshape((n,1))
    res = y[i] - b_s[i]
    norm_ai = np.power(np.linalg.norm(a_i),2)
    weighted_projection_sum = (res[0]/(b_norm*norm_ai))*a_i
    #print("x_s1",x_s)
    alpha = 1
    x_update = alpha*weighted_projection_sum
    #print(x_update)
    x_s = x_s - x_update.reshape(x_s.shape)

    # x_s_max = max(x_s)
    # x_s_min = min(x_s)
    # x_s = (x_s-x_s_min)/(x_s_max - x_s_min)
    #print(x_s)
    return x_s,alpha

def globalBlockRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,row_size,start,MAX_ALPHA,ALPHA_MULT,alpha=-1):

    '''
    This is a coarser and possibly incorrect implementation of Theorem 4.1 of paper https://arxiv.org/pdf/1902.09946.pdf
    '''

    weighted_projection_sum = np.zeros((n, 1))

    for i in range(row_size):
        w_bars_i = w_bars[i][0]
        a_i = scaled_A[i].reshape((n,1))
        res = y[i] - b_s[i]

        weighted_projection_sum = weighted_projection_sum + w_bars_i * (res[0])*a_i
    alpha = float(1)/row_size
    x_update = alpha*weighted_projection_sum
    x_s = x_s - x_update.reshape(x_s.shape)
    return x_s,alpha

def globalFastBlockRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,row_size,start,b_norm):

    '''
    implementation attempt of Theorem 4.2 from paper https://arxiv.org/pdf/1902.09946.pdf
    some minor changes have been made such that alpha value is capped at MAX_ALPHA,
    Alpha values are also scaled with ALPHA_MULT
    '''
    weighted_projection_sum = np.zeros((n,1))
    weighted_res_sum = 0 #np.zeros((row_size,1))
    for i in range(row_size):
        w_bars_i = w_bars[i][0] #.reshape((1,1))
        a_i = scaled_A[i].reshape((n,1))
        res = (y[i] - b_s[i])/b_norm

        weighted_projection_sum = weighted_projection_sum + w_bars_i * (res[0])*a_i
        #print(weighted_projection_sum)
        weighted_res_sum = weighted_res_sum + w_bars_i * (res[0])*(res[0])


    alpha_num = weighted_res_sum
    alpha_denom = np.power(np.linalg.norm(weighted_projection_sum),2)
    alpha = 1.0 * alpha_num / alpha_denom
    # if alpha == -1:
    #     alpha = ALPHA_MULT*alpha_num/alpha_denom
    #     # if alpha>MAX_ALPHA:
    #     #     alpha=MAX_ALPHA

    x_update = alpha*weighted_projection_sum
    x_s = x_s - x_update.reshape(x_s.shape)

    print(alpha)
    return x_s,alpha

def updateProbabilities(row,x,x_list,curr_norm_val,norm_val,trials,failures,probabilities):
    #curr_norm_val = np.linalg.norm(x - x_true)
    minval, key = min((v, i) for i, v in enumerate(norm_val))
    sigma = np.std(norm_val)
    mu = np.mean(norm_val)
    lim = 5 * sigma
    if curr_norm_val > mu + lim or curr_norm_val < mu - lim:
        x = x_list[key]
        curr_norm_val = norm_val[key]
        failures[row] += 1
    else:
        for i in range(m):
            if x[i] > 1 or x[i] < 0:
                x = x_list[key]
                curr_norm_val = norm_val[key]
                failures[row] += 1
                break

    trials[row] += 1
    probabilities[row] *= (1.0 - float(failures[row]) / trials[row])

    sum_p = sum(probabilities)
    probabilities = [p / sum_p for p in probabilities]

    return x,curr_norm_val,probabilities

comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


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

col_parts = 8
row_parts = 8

m=32
n=32

row_p_size = int(m/row_parts)
col_p_size = int(n/col_parts)

MAX_ITR = 500

ROOT_PROCESS_RANK=size-1
turnOnHardware = 1
blockRK = False
MAX_TOL = 1.0
MIN_TOL = 0.0

if rank == ROOT_PROCESS_RANK:


    #obtain an A matrix with values between 0,1
    #I have observed that having matrix between 0,1 gives the best results
    scaled_A = np.random.randint(0,10000,size=(m,n))/10000.0
    scaled_A_row_norm = np.zeros((m,1))
    w_bars = np.zeros((m,1))
    scaled_A_frobenius_rows = np.zeros((m, 1))
    for i in range(m):
        scaled_A_row_norm[i] = np.linalg.norm(scaled_A[i][:])
        scaled_A[i,:] = scaled_A[i,:]/scaled_A_row_norm[i]
        w_bars[i] = float(1) / (row_p_size*np.power(scaled_A_row_norm[i], 2))
        scaled_A_frobenius_rows[i][0] = np.linalg.norm(scaled_A[i,:])


    x_true = np.random.rand(n,1)

    b = np.dot(scaled_A,x_true)
    b_norm = np.linalg.norm(b)
    # b=b/b_norm

    norm_val = []

    scatter_A_matrix = []

    scaled_A_frobenius = np.linalg.norm(scaled_A,ord='fro')
    scaled_A_frobenius_parts = np.zeros((row_parts, 1))


    for r in range(row_parts):
        row_start = row_p_size * r
        row_end = row_p_size * r + row_p_size
        for c in range(col_parts):
            col_start = col_p_size*c
            col_end = col_p_size*c+col_p_size

            #print(row_start,row_end,col_start,col_end,scaled_A[row_start:row_end,col_start:col_end] )
            scatter_A_matrix.append(scaled_A[row_start:row_end,col_start:col_end])

        scaled_A_frobenius_parts[r][0] = np.linalg.norm(scaled_A[row_start:row_end,:],ord='fro')

    if blockRK:
        size = row_parts
        failures = [0 for i in range(size)]
        trials = [0 for i in range(size)]
        probabilities = [np.power(scaled_A_frobenius_parts[i][0] / scaled_A_frobenius, 2) for i in
                               range(size)]
    else:
        size = m
        failures = [0 for i in range(size)]
        trials = [0 for i in range(size)]
        probabilities = [np.power(scaled_A_frobenius_rows[i][0] / scaled_A_frobenius, 2) for i in range(size)]

    scatter_A_matrix.append(None)

    scaled_A_recv = comm.scatter(scatter_A_matrix,root=ROOT_PROCESS_RANK)
    k = 0

    x = np.zeros(n)
    #x = np.random.rand(n)
    x_list = []
    alpha_list = []

    MAX_ALPHA = 1
    ALPHA_MULT = 0.5

    while k < MAX_ITR:

        #choose random row of processors
        if blockRK:
            #row_id =  np.random.randint(0,row_parts)
            row_id = np.random.choice(np.arange(0, row_parts), p=probabilities)
        else:
            row = np.random.choice(np.arange(0, m), p=probabilities)
            row_id = int(row/row_p_size)

        data = np.array([row_id], dtype='i')
        comm.Bcast(data, root=ROOT_PROCESS_RANK)

        sum_y = parallelMatVec(x)

        start = row_id * row_p_size
        end = (row_id + 1) * row_p_size

        b_s = b[start:end]

        w_bars_s = w_bars[start:end]

        scaled_A_s = scaled_A[start:end, :]

        #x_s, alpha = globalRandomizedKaczmarz(sum_y, x, b_s, w_bars_s, scaled_A_s, row_p_size,row-start,b_norm)
        #x_s,alpha = globalBlockRandomizedKaczmarz(sum_y,x,b_s,w_bars_s,scaled_A_s,row_p_size,start,MAX_ALPHA,ALPHA_MULT,alpha_val)
        if blockRK:
            x_s, alpha = globalFastBlockRandomizedKaczmarz(sum_y, x, b_s, w_bars_s, scaled_A_s, row_p_size, start, b_norm)
        else:
            x_s, alpha = globalRandomizedKaczmarz(sum_y, x, b_s, w_bars_s, scaled_A_s, row_p_size, row - start, b_norm)

        x = x_s.reshape((n,1))

        curr_norm_val = np.linalg.norm(x-x_true)/np.linalg.norm(x_true)

        if k>1:
            if blockRK:
                x,curr_norm_val,probabilities = updateProbabilities(row_id,x,x_list,curr_norm_val,norm_val,trials,failures,probabilities)
            else:
                x,curr_norm_val,probabilities = updateProbabilities(row, x, x_list, curr_norm_val, norm_val, trials, failures,
                                    probabilities)

        norm_val.append(curr_norm_val)
        x_list.append(x)
        alpha_list.append(alpha)

        err = x - x_true
        k = k + 1

    for i in range(n):
        print(k, x_true[i], x[i])

    print(norm_val)
    fig1 = plt.plot(norm_val)
    plt.savefig("./plots/norm_val.png")
    #print(alpha_list)


else:
    k=0
    scatter_A_matrix = None
    scaled_A = comm.scatter(scatter_A_matrix, root=ROOT_PROCESS_RANK)

    print("Process:",rank,scaled_A)

    meliso_obj = meliso.MelisoPy(1,row_p_size,col_p_size,MAX_TOL,MIN_TOL,turnOnHardware)

    #initialize weights to 0 on the memristor device
    meliso_obj.initializeWeights()

    #set weights on the memristor
    meliso_obj.setWeights(scaled_A)

    print("Initialized submatrix on process {}".format(rank))

    my_row_rank = int(rank/col_parts)
    my_col_rank = int(rank-my_row_rank*col_parts)

    while k < MAX_ITR:
        data = np.empty(1, dtype='i')
        comm.Bcast(data, root=ROOT_PROCESS_RANK)

        if data[0] == my_row_rank:

            x = np.empty(col_p_size, dtype=np.float64)

            comm.Recv(x,source=ROOT_PROCESS_RANK)
            meliso_obj.loadInput(x)
            meliso_obj.matVec()
            y = meliso_obj.getResults()

            comm.Send(y, dest=ROOT_PROCESS_RANK)

        k = k + 1
