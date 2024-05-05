from solver.matvec.MatVecSolver import MatVecSolver
import numpy as np

'''
Update as of 5/3/24: This code works but doesnt really converge/has weak convergence.
For the bcsstk02 matrix, the convergence is very hard especially
Maybe the 0,1 scaling of both A is not working well.
This can cause the condition number of the matrix to really shoot up, causing poor convergence
One idea could be to impose hard constraints regarding potential values that each matrix element could possibly take.
Update as of 5/4/24: It is the condition no of the matrix, post scaling that makes it super hard to solve
A random matrix is also giving a condition no of 28k post 0-1 scaling.
'''

def updateProbabilities(row,x,x_list,curr_norm_val,norm_val,trials,failures,probabilities):
    #curr_norm_val = np.linalg.norm(x - x_true)
    minval, key = min((v, i) for i, v in enumerate(norm_val))
    sigma = np.std(norm_val)
    mu = np.mean(norm_val)
    lim = 3 * sigma
    if curr_norm_val > mu + lim or curr_norm_val < mu - lim:
        x = x_list[key]
        curr_norm_val = norm_val[key]
        failures[row] = failures[row] + 1
    else:
        for i in range(m):
            if x[i] > 1.0 or x[i] < 0:
                x = x_list[key]
                curr_norm_val = norm_val[key]
                failures[row] = failures[row] + 1
                break

    trials[row] = trials[row] + 1
    old_probabilities = probabilities.copy()
    probabilities[row] *= ( 1.0 - float(failures[row]) / trials[row])

    print(trials,failures,probabilities)

    sum_p = sum(probabilities)
    if sum_p == 0:
        probabilities = old_probabilities
    else:
        probabilities = [p/sum_p for p in probabilities]

    return x,curr_norm_val,probabilities

def globalBlockCoordinateDescent(y,x_s,b_s,w_bars,scaled_A,er,sr,row_size,b_norm,n):
    '''
    This is the classical Block Randomized Kaczmarz. Theorem 4.1 of https://arxiv.org/pdf/1902.09946
    '''
    row_size = er - sr
    weighted_projection_sum = np.zeros((n, 1))
    normsum = 0
    for i in range(row_size):
        w_bars_i = w_bars[i][0]  # .reshape((1,1))
        a_i = scaled_A[i].reshape((n, 1))
        a_i_norm2 = np.power(np.linalg.norm(a_i),2)
        normsum = normsum + a_i_norm2
        res = (y[i] - b_s[i]) #/ a_i_norm2
        # print(res)
        weighted_projection_sum = weighted_projection_sum + (res) * a_i

        #weighted_res_sum = weighted_res_sum + w_bars_i * (res) * (res)
    # print(weighted_res_sum)
    alpha = 1
    x_update = alpha * weighted_projection_sum/normsum
    x_s = x_s - x_update.reshape(x_s.shape)
    return x_s, alpha


def globalBlockRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,er,sr,row_size,b_norm,n):
    '''
    This is the classical Block Randomized Kaczmarz. Theorem 4.1 of https://arxiv.org/pdf/1902.09946
    '''
    row_size = er - sr
    weighted_projection_sum = np.zeros((n, 1))
    for i in range(row_size):
        w_bars_i = w_bars[i][0]  # .reshape((1,1))
        a_i = scaled_A[i].reshape((n, 1))
        a_i_norm2 = np.power(np.linalg.norm(a_i),2)
        res = (y[i] - b_s[i]) / a_i_norm2
        # print(res)
        weighted_projection_sum = weighted_projection_sum + (res) * a_i

        #weighted_res_sum = weighted_res_sum + w_bars_i * (res) * (res)
    # print(weighted_res_sum)
    alpha = 1/(er-sr)
    x_update = alpha * weighted_projection_sum
    x_s = x_s + x_update.reshape(x_s.shape)
    return x_s, alpha

def globalFastBlockRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,er,sr,row_size,b_norm,n):

    '''
    implementation attempt of Theorem 4.2 from paper https://arxiv.org/pdf/1902.09946.pdf
    some minor changes have been made such that alpha value is capped at MAX_ALPHA,
    Alpha values are also scaled with ALPHA_MULT
    '''

    weighted_projection_sum = np.zeros((n,1))
    weighted_res_sum = 0 #np.zeros((row_size,1))
    row_size = er-sr

    for i in range(row_size):
        w_bars_i = w_bars[i][0] #.reshape((1,1))
        a_i = scaled_A[i].reshape((n,1))
        res = (y[i] - b_s[i])/b_norm
        #print(res)
        weighted_projection_sum = weighted_projection_sum + w_bars_i * (res)*a_i

        weighted_res_sum = weighted_res_sum + w_bars_i * (res)*(res)
    #print(weighted_res_sum)

    alpha_num = weighted_res_sum
    alpha_denom = np.power(np.linalg.norm(weighted_projection_sum),2)
    alpha = 1e-4 * alpha_num / alpha_denom
    x_update = alpha*weighted_projection_sum
    x_s = x_s + x_update.reshape(x_s.shape)

    # print(x_s.shape,weighted_projection_sum.shape,n)
    # print(np.max(scaled_A))
    # print(np.min(scaled_A))
    # print(w_bars[2][0])
    # print(w_bars)
    # print(b_norm)
    # print(weighted_res_sum)
    # print(weighted_projection_sum)
    # print(alpha)
    return x_s,alpha

def initRK(scaled_A,virtualizer,row_parts,row_part_size):
    m = scaled_A.shape[0]
    scaled_A_frobenius = np.linalg.norm(scaled_A, ord='fro')
    scaled_A_row_norm = np.zeros((m, 1))
    w_bars = np.zeros((m, 1))
    scaled_A_frobenius_rows = np.zeros((m, 1))
    scaled_A_frobenius_parts = np.zeros((row_parts, 1))
    for i in range(m):
        scaled_A_row_norm[i] = np.linalg.norm(scaled_A[i][:])
        #scaled_A[i, :] = scaled_A[i, :] / scaled_A_row_norm[i]
        w_bars[i] = float(1) / (row_part_size * np.power(scaled_A_row_norm[i], 2))
        scaled_A_frobenius_rows[i][0] = np.linalg.norm(scaled_A[i, :])

    #print(np.sum(scaled_A_row_norm,axis=0),scaled_A_frobenius,)
    for r in range(row_parts):
        sr = virtualizer[r, 0]["rc_limits"][0][0]
        er = virtualizer[r, 0]["rc_limits"][0][1]
        #print(sr,er,scaled_A_frobenius,sr,er)

        scaled_A_frobenius_parts[r][0] = np.linalg.norm(scaled_A[sr:er][:], ord='fro')
        #print(sr, er, scaled_A_frobenius, scaled_A_frobenius_parts[r][0], scaled_A[sr:er, :].shape)

    probabilities = [np.power(scaled_A_frobenius_parts[i][0] / scaled_A_frobenius, 2) for i in
                     range(row_parts)]

    failures = [0 for i in range(row_parts)]
    trials = [0 for i in range(row_parts)]


    return np.copy(scaled_A),scaled_A_frobenius,scaled_A_row_norm,w_bars,scaled_A_frobenius_rows,scaled_A_frobenius_parts,probabilities,trials,failures

def correctY(n,y,a_min,a_max,a_row_sum,x_min,x_max,x_sum):
    correctedY = np.copy(y)
    for i in range(y.shape[0]):
        correctedY[i] = correctedY[i]*(a_max*x_max) + a_min*x_sum + x_min*a_row_sum[i] - n*a_min*x_min
        #correctedY[i] = correctedY[i] * (1 * x_max) + x_min * a_row_sum[i]

    return correctedY

def rootSolve():

    mv = MatVecSolver()
    #mv.solverObject.initializeMat(np.random.rand(128, 64))
    # #mv.solverObject.initializeMat(2*np.random.rand(128, 128))
    #mv.solverObject.initializeX(np.loadtxt(fname="input_x", delimiter=','))

    real_x_true = np.copy(mv.solverObject.x)

    # a_min = mv.solverObject.mca.mat_min
    # a_max = mv.solverObject.mca.mat_max
    # a_row_sum = mv.solverObject.mca.mat_row_sum

    scaled_A, \
    scaled_A_frobenius, \
    scaled_A_row_norm, \
    w_bars, \
    scaled_A_frobenius_rows, \
    scaled_A_frobenius_parts, \
    probabilities, \
    trials, \
    failures    = initRK(mv.solverObject.origMat,
                           mv.solverObject.virtualizer,
                           mv.solverObject.maxVRows,
                           mv.solverObject.mcaGridRowCap)

    cond_no =np.linalg.cond(scaled_A)

    mv.parallelizedBenchmarkMatVec(0, 0)

    mv.finalize()
    b= mv.solverObject.y_benchmark_result
    # b = correctY(mv.solverObject.maxVCols,
    #          mv.solverObject.y_benchmark_result,
    #          a_min,
    #          a_max,
    #          a_row_sum,
    #          mv.solverObject.x_min,
    #          mv.solverObject.x_max,
    #          mv.solverObject.x_sum)

    #print(b, a_min, a_max, a_row_sum, mv.solverObject.x_min, mv.solverObject.x_max, mv.solverObject.x_sum, )

    b_norm = np.linalg.norm(b)

    x = np.random.randn(*mv.solverObject.x.shape) #1e-6*np.random.rand(mv.solverObject.x,1) #np.copy(mv.solverObject.x)
    n = x.shape[0]
    y = np.zeros(mv.solverObject.origMatRows, dtype=np.float64)

    x_list = []
    norm_val = []
    alpha_list = []

    for k in range(100):

        mv.solverObject.initializeX(x)

        #implement row selection
        i = np.random.choice(np.arange(0, mv.solverObject.maxVRows), p=probabilities) #np.random.randint(0,mv.solverObject.maxVRows)

        for j in range(mv.solverObject.maxVCols):
            mv.solverObject.virtualParallelMatVec( i, j)

        sr = mv.solverObject.virtualizer[i, 0]["rc_limits"][0][0]
        er = mv.solverObject.virtualizer[i, 0]["rc_limits"][0][1]
        y[sr:er] = np.copy(mv.solverObject.virtualizer[i]["y"])
        mv.solverObject.virtualizer[i]["y"] = np.zeros(er - sr, dtype=np.float64)

        # #TODO: Obtain true y based on rescaling back output
        # y = correctY(mv.solverObject.maxVCols,
        #              y,
        #              a_min,
        #              a_max,
        #              a_row_sum,
        #              mv.solverObject.x_min,
        #              mv.solverObject.x_max,
        #              mv.solverObject.x_sum)

        #print(b, a_min, a_max, a_row_sum, mv.solverObject.x_min, mv.solverObject.x_max, mv.solverObject.x_sum, )

        b_s = b[sr:er]

        w_bars_s = w_bars[sr:er]

        scaled_A_s = scaled_A[sr:er, :]

        #x_s, alpha = globalFastBlockRandomizedKaczmarz(y, x, b_s, w_bars_s, scaled_A_s, er, sr, mv.solverObject.mcaGridRowCap, b_norm, n)
        #x_s, alpha = globalBlockRandomizedKaczmarz(y, x, b_s, w_bars_s, scaled_A_s, er, sr,mv.solverObject.mcaGridRowCap, b_norm, n)
        x_s, alpha = globalBlockCoordinateDescent(y, x, b_s, w_bars_s, scaled_A_s,er,sr, mv.solverObject.mcaGridRowCap, b_norm,n)



        x = np.copy(x_s.reshape((n, 1)))

        #x = x.reshape((n, 1)) + x_s.reshape((n, 1))

        curr_norm_val = np.linalg.norm(x - real_x_true) / np.linalg.norm(real_x_true)

        # x, curr_norm_val, probabilities = updateProbabilities(i, x, x_list, curr_norm_val, norm_val, trials,
        #                                                       failures, probabilities)

        x_list.append(x)
        norm_val.append(curr_norm_val)
        alpha_list.append(alpha)
        # print(x)
        # print(np.linalg.norm(x - real_x_true))
        # print(np.linalg.norm(real_x_true))
        print(alpha_list)
        print(norm_val)
        print(cond_no)
        #use to implement other aspects of Kaczmarz

        #a new A matrix can be re-initialized using:
        #mv.solverObject.initializeMatrix(new_mat)

        #initialize a new x using:
        #mv.solverObject.initializeX(new_x)

        #Remember!: always initialize new matrix before initializing a new vector (if your new matrix has different dimensions)

    mv.finalize()
