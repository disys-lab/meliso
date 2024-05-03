from solver.matvec.MatVecSolver import MatVecSolver

def globalFastBlockRandomizedKaczmarz(y,x_s,b_s,w_bars,scaled_A,row_size,b_norm):

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
    x_update = alpha*weighted_projection_sum
    x_s = x_s - x_update.reshape(x_s.shape)

    print(alpha)
    return x_s,alpha

def initRK(scaled_A,virtualizer,row_parts,row_part_size):
    m = scaled_A.shape[0]
    scaled_A_frobenius = np.linalg.norm(scaled_A, ord='fro')
    scaled_A_row_norm = np.zeros((m, 1))
    w_bars = np.zeros((m, 1))
    scaled_A_frobenius_rows = np.zeros((m, 1))
    for i in range(m):
        scaled_A_row_norm[i] = np.linalg.norm(scaled_A[i][:])
        scaled_A[i, :] = scaled_A[i, :] / scaled_A_row_norm[i]
        w_bars[i] = float(1) / (row_part_size * np.power(scaled_A_row_norm[i], 2))
        scaled_A_frobenius_rows[i][0] = np.linalg.norm(scaled_A[i, :])


    for r in range(row_parts):
        sr = virtualizer[r, 0]["rc_limits"][0][0]
        er = virtualizer[r, 0]["rc_limits"][0][1]
        scaled_A_frobenius_parts[r][0] = np.linalg.norm(scaled_A[sr:er, :], ord='fro')

    probabilities = [np.power(scaled_A_frobenius_parts[i][0] / scaled_A_frobenius, 2) for i in
                     range(row_parts)]


    return np.copy(scaled_A),scaled_A_frobenius,scaled_A_row_norm,w_bars,scaled_A_frobenius_rows,scaled_A_frobenius_parts,probabilities

def correctY(n,y,a_min,a_max,a_row_sum,x_min,x_max,x_sum):
    correctedY = np.copy(y)
    for i in range(y.shape[0]):
        correctedY[i] = correctedY[i]*(a_max*x_max) + a_min*x_sum + x_min*a_row_sum[i] - n*a_min*x_min

    return correctedY

def rootSolve():

    mv = MatVecSolver()

    a_min = mv.solverObject.mca.mat_min
    a_max = mv.solverObject.mca.mat_max
    a_row_sum = mv.solverObject.mca.mat_row_sum

    scaled_A, \
    scaled_A_frobenius, \
    scaled_A_row_norm, \
    w_bars, \
    scaled_A_frobenius_rows, \
    scaled_A_frobenius_parts, \
    probabilities = initRK(mv.solverObject.origMat,
                           mv.solverObject.virtualizer,
                           mv.solverObject.maxVRows,
                           mv.solverObject.mcaGridRowCap)

    mv.parallelizedBenchmarkMatVec(0, 0)

    mv.finalize()

    b = mv.y_benchmark_result
    b_norm = np.linalg.norm(b)

    x = np.copy(mv.solverObject.x)
    y = np.zeros(self.origMatRows, dtype=np.float64)

    for k in range(10):

        mv.solverObject.initializeX(x)

        #implement row selection
        i = np.random.choice(np.arange(0, mv.solverObject.maxVRows), p=probabilities) #np.random.randint(0,mv.solverObject.maxVRows)

        for j in range(mv.solverObject.maxVCols):
            mv.solverObject.virtualParallelMatVec( i, j)

        sr = self.virtualizer[i, 0]["rc_limits"][0][0]
        er = self.virtualizer[i, 0]["rc_limits"][0][1]
        y[sr:er] = np.copy(mv.solverObject.virtualizer[i]["y"])
        self.virtualizer[i]["y"] = np.zeros(er - sr, dtype=np.float64)

        #TODO: Obtain true y based on rescaling back output
        y = correctY(mv.solverObject.maxVCols,
                     y,
                     a_min,
                     a_max,
                     a_row_sum,
                     mv.solverObject.x_min,
                     mv.solverObject.x_max,
                     mv.solverObject.x_sum)

        b_s = b[sr:er]

        w_bars_s = w_bars[sr:er]

        scaled_A_s = scaled_A[sr:er, :]

        x_s, alpha = globalFastBlockRandomizedKaczmarz(y, x, b_s, w_bars_s, scaled_A_s, mv.solverObject.mcaGridRowCap, b_norm)

        x = x.reshape((n, 1)) + x_s.reshape((n, 1))

        #use to implement other aspects of Kaczmarz

        #a new A matrix can be re-initialized using:
        #mv.solverObject.initializeMatrix(new_mat)

        #initialize a new x using:
        #mv.solverObject.initializeX(new_x)

        #Remember!: always initialize new matrix before initializing a new vector (if your new matrix has different dimensions)

    mv.finalize()
