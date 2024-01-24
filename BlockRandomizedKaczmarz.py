import meliso
import numpy as np
from mpi4py import MPI

def parallelMatVec(x):
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

def globalRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,row_size,start):
    weighted_projection_sum = np.zeros((n,1))
    weighted_res_sum = 0 #np.zeros((row_size,1))
    for i in range(row_size):
        w_bars_i = w_bars[i][0] #.reshape((1,1))
        a_i = scaled_A[i].reshape((n,1))
        res = y[i] - b_s[i]

        weighted_projection_sum = weighted_projection_sum + w_bars_i * (res[0])*a_i
        #print(weighted_projection_sum)
        weighted_res_sum = weighted_res_sum + w_bars_i * (res[0])*(res[0])


    alpha_num = weighted_res_sum
    alpha_denom = np.power(np.linalg.norm(weighted_projection_sum),2)

    alpha = 1 #alpha_num/alpha_denom #0.1 works really well

    x_update = alpha*weighted_projection_sum
    x_s = x_s - x_update.reshape(x_s.shape)

    print(alpha)
    return x_s



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

col_parts = 4
row_parts = 4

m=32
n=32

row_p_size = int(m/row_parts)
col_p_size = int(n/col_parts)

MAX_ITR = 100

ROOT_PROCESS_RANK=size-1
turnOnHardware = False

if rank == ROOT_PROCESS_RANK:


    #obtain an A matrix with values between 0,1
    #I have observed that having matrix between 0,1 gives the best results
    scaled_A = np.random.randint(0,1000,size=(m,n))/1000.0
    scaled_A_row_norm = np.zeros((m,1))
    w_bars = np.zeros((m,1))
    for i in range(m):
        scaled_A_row_norm[i] = np.linalg.norm(scaled_A[i][:])
        scaled_A[i][:] = scaled_A[i][:]/scaled_A_row_norm[i][0]
        w_bars[i] = 1/(row_p_size * np.power(scaled_A_row_norm[i], 2))

    #print(scaled_A_row_norm)



    #print(w_bars)
    # w_bars = float(1.0/(row_p_size*np.power(scaled_A_row_norm,2)))

    x_true = np.random.rand(n,1)

    b = np.dot(scaled_A,x_true)



    norm_val = []

    scatter_A_matrix = []

    for r in range(row_parts):
        for c in range(col_parts):

            row_start = row_p_size*r
            row_end = row_p_size*r+row_p_size

            col_start = col_p_size*c
            col_end = col_p_size*c+col_p_size

            #print(row_start,row_end,col_start,col_end,scaled_A[row_start:row_end,col_start:col_end] )
            scatter_A_matrix.append(scaled_A[row_start:row_end,col_start:col_end])

    scatter_A_matrix.append(None)

    scaled_A_recv = comm.scatter(scatter_A_matrix,root=ROOT_PROCESS_RANK)
    k = 0

    x = np.zeros(n)

    while k < MAX_ITR:

        #choose random row of processors
        row_id =  np.random.randint(0,row_parts)
        data = np.array([row_id], dtype='i')
        comm.Bcast(data, root=ROOT_PROCESS_RANK)

        sum_y = parallelMatVec(x)
        # print(sum_y)
        # print(b)

        start = row_id * row_p_size
        end = (row_id + 1) * row_p_size

        b_s = b[start:end]

        w_bars_s = w_bars[start:end]

        scaled_A_s = scaled_A[start:end, :]

        x_s = globalRandomizedKaczmarz(sum_y,x,b_s,w_bars_s,scaled_A_s,row_p_size,start)
        x = x_s.reshape((n,))

        #print("Itr:{}, norm:{}".format(k,np.linalg.norm(x-x_true)))
        norm_val.append(np.linalg.norm(x-x_true))
        k = k + 1

    print(norm_val)

else:
    k=0
    scatter_A_matrix = None
    scaled_A = comm.scatter(scatter_A_matrix, root=ROOT_PROCESS_RANK)

    print("Process:",rank,scaled_A)

    meliso_obj = meliso.MelisoPy(3,row_p_size,col_p_size,0,turnOnHardware)

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



#
#     #choose a partition at random
#     part_k = np.random.randint(0,parts)
#
#     start = p_size*(part_k)
#     end = start + p_size
#
#     x_s = x
#     b_s = b[start:end][:]
#     #consider the actual input vector and get hardware matvec results
#
#     meliso_obj_list[part_k].loadInput(x_s)
#     meliso_obj_list[part_k].matVec()
#     y = meliso_obj_list[part_k].getResults()
#
#     weighted_projection_sum = np.zeros((n,1))
#     weighted_res_sum = np.zeros((p_size,1))
#     for i in range(p_size):
#         w_bars_i = w_bars[start + i].reshape((1,1))
#         a_i = scaled_A[i].reshape((n,1))
#         weighted_projection_sum = weighted_projection_sum + w_bars_i * (y[i] - b_s[i])*a_i
#         weighted_res_sum[i] = w_bars_i * (y[i] - b_s[i])
#
#     alpha_num = np.sum(weighted_res_sum)
#     alpha_denom = np.power(np.linalg.norm(weighted_projection_sum),2)
#
#     alpha = alpha_num/alpha_denom
#     x_s = x_s +alpha*weighted_projection_sum
#
#     #real_Ax = np.dot(scaled_A,x)
#
#
#     #print("y_rescaled:",y_rescaled_mem_result.reshape((1,32)))
#     #print("real_Ax:",real_Ax.reshape((1,32)))
#     #print(y_rescaled_mem_result.reshape((1,32))-real_Ax.reshape((1,32)))
#     k = k+1
#

