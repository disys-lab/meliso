import meliso
import numpy as np
from scipy.io import mmread

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

# p=4
# f = np.zeros((1,4)).flatten()
# d = np.array([12.0,13.0,14.0,16.0])
# t = np.array([5.0,6.0,9.0,11.0])
#
# computeInterpolants(p-1,0,p-1,f,d,t,1)
#
# value = 7.0
#
# res = evaluatePolynomial(p-1,value,f,t)
#
# print(res)

def matVec(meliso_obj,x,RESULT_MULT):
    meliso_obj.loadInput(x)
    meliso_obj.matVec()
    y_rescaled_mem_result = RESULT_MULT * meliso_obj.getResults()

    return y_rescaled_mem_result

def setWeightsIncremental(meliso_obj,scaled_A,res_tol,precision,itr_limit):
    # set weights on the memristor
    j = 0
    res=0
    while j<itr_limit:
        meliso_obj.setWeightsIncremental(scaled_A, precision)
        actualWeights = meliso_obj.getWeights()
        curr_res = np.linalg.norm(actualWeights - scaled_A)
        if abs(res-curr_res)<res_tol and j>0:
            res = curr_res
            print("INFO:setWeights::", j, res, curr_res)
            break
        res = curr_res
        j = j + 1

    return j,res
    # actualWeights = meliso_obj.getWeights()
    # print(actualWeights)
    # print(scaled_A)

def correctionADMMDenoising(meliso_obj,A,x,RESULT_MULT,res_tol,precision,itr_limit,tolerance,y_benchmark,benchmark_norm):
    # l_dn = 0.001
    # rho = 0.001
    # ifac = 2
    # eta = 1e-3

    # l_dn = 0.001
    # rho = 0.001
    # ifac = 4
    # eta = 1e-3

    l_dn = 0.0001
    rho = 0.0001
    ifac = 1
    eta = 1e-3

    ILIM = itr_limit / (ifac)
    OLIM = ifac

    alpha_x = 1
    alpha_a = 1

    samples = int(dim*0.1)

    rows = A.shape[0]
    cols = A.shape[1]

    L = np.eye(rows)
    for i in range(rows - 1):
        L[i, i + 1] = -1

    lbda = np.zeros((rows, 1), dtype=float).flatten()

    LTL = L.T @ L

    xt = x.reshape((1,cols))
    X = np.tile(xt,(rows,1))

    y_a = np.zeros((rows, 1), dtype=float)

    y_x = np.zeros((rows, 1), dtype=float)

    U_tilde = np.empty((rows, 1), dtype=float)

    V_tilde_a = np.empty((rows, 1), dtype=float)

    V_tilde_x = np.empty((rows, 1), dtype=float)

    A_tilde = np.copy(A)
    X_tilde = np.copy(X)
    y_corr = 0
    X_itrs = 0
    A_itrs = 0
    A_res = 0
    X_res = 0

    y_corr_previous= np.copy(y_a)
    y_corr_norm_list = []
    dual_norm_diff_list = []
    benchmark_err_norm_list = []

    for itr in range(OLIM):

        X_j, X_res = setWeightsIncremental(meliso_obj, X_tilde, res_tol, precision, alpha_x * ILIM)

        X_itrs = X_itrs + X_j

        X_tilde = meliso_obj.getWeights()

        for i in range(rows):
            ai = A[i, :].flatten()
            ui_tilde = matVec(meliso_obj, ai, RESULT_MULT).flatten()
            ui_tilde = np.linalg.solve(np.eye(rows) + l_dn * LTL, ui_tilde)
            U_tilde[i] = ui_tilde[i]

        for i in range(rows):
            ait = A_tilde[i, :].flatten()
            vi_tilde = matVec(meliso_obj, ait, RESULT_MULT)
            vi_tilde = np.linalg.solve(np.eye(rows) + l_dn * LTL, vi_tilde)
            V_tilde_x[i] = vi_tilde[i]

        # for i in range(rows):
        for s in range(samples):
            r = np.random.randint(low=0, high=rows - 1)
            gradX = 2 * np.dot(X_tilde[r, :] ,(X_tilde[r, :] - X[r, :])) - (lbda[r] + rho * (y_x[r] - y_a[r])) * (A_tilde[r, :] - A[r, :])  # (-lbda[i] + rho * U_tilde[i]) * A[i,:]
            X_tilde[r, :] = X_tilde[r, :] + eta * gradX

        # Optimize for A
        A_j, A_res = setWeightsIncremental(meliso_obj, A_tilde, res_tol, precision, alpha_a * ILIM)

        A_itrs = A_itrs + A_j

        A_tilde = meliso_obj.getWeights()

        for i in range(rows):
            xit = X_tilde[i,:].flatten()
            vi_tilde = matVec(meliso_obj,xit,RESULT_MULT)
            vi_tilde = np.linalg.solve(np.eye(rows) + l_dn * LTL, vi_tilde)
            V_tilde_a[i] = vi_tilde[i]

        for s in range(samples):
            r = np.random.randint(low=0, high=rows - 1)
            gradA = 2 * np.dot(A_tilde[r, :], (A_tilde[r, :] - A[r, :])) + (lbda[r] + rho * (y_x[r] - y_a[r])) * (X_tilde[r, :] - X[r, :])  # (-lbda[i] + rho * U_tilde[i]) * A[i,:]
            A_tilde[r, :] = A_tilde[r, :] + eta * gradA

        y_tilde = matVec(meliso_obj,x,RESULT_MULT)

        y_tilde = np.linalg.solve(np.eye(rows) + l_dn * LTL, y_tilde)


        for i in range(rows):
            y_a[i] = y_tilde[i] - V_tilde_a[i] + U_tilde[i] #+ (DAX_tilde[i] + DXA_tilde[i])/2.0
            y_x[i] = U_tilde[i] - V_tilde_x[i] + y_tilde[i]

            #print(y_a[i]+y_x[i],y_tilde[i],V_tilde_a[i],V_tilde_x[i],U_tilde[i])

        y_a = np.linalg.solve(np.eye(rows) + l_dn * LTL, y_a)
        y_x = np.linalg.solve(np.eye(rows) + l_dn * LTL, y_x)

        for i in range(rows):
            lbda[i] = lbda[i] + rho * (y_x[i].flatten() - y_a[i].flatten())

        y_corr = 0.5*(y_a+y_x)

        y_corr = np.linalg.solve(np.eye(rows)+lbda*LTL, y_corr)
        dual_norm_diff = np.linalg.norm(y_a-y_x,ord=np.inf)
        dual_norm_diff_list.append(dual_norm_diff)

        y_nrm = np.linalg.norm(y_corr - y_corr_previous,ord=np.inf)
        y_corr_norm_list.append(y_nrm)
        y_corr_previous = np.copy(y_corr)

        benchmark_diff_norm = np.linalg.norm(y_corr.flatten() - y_benchmark.flatten(),ord=np.inf)

        benchmark_err_norm_list.append(benchmark_diff_norm)

        if benchmark_diff_norm <= benchmark_norm:
            break

        if dual_norm_diff <= tolerance and y_nrm <= tolerance:
            break

    #print(y_corr_norm_list)
    print(dual_norm_diff_list)
    print(benchmark_err_norm_list)
    return y_corr,[X_itrs,0,0],A_itrs,X_res,A_res

MAX_TOL = 1.0
MIN_TOL = 0

p=10


#scaled_A = np.loadtxt(fname='matrices/3232_random.mtx',delimiter=',')
scaled_A = mmread('matrices/3232_random.mtx')
#scaled_A = mmread('matrices/bcsstk02.mtx')

if not isinstance(scaled_A, np.ndarray):
    scaled_A = scaled_A.toarray()
    print(scaled_A)
    scaled_A -= scaled_A.min()
    scaled_A /= scaled_A.ptp()
    # scaled_A = 2.0*scaled_A - 1.0

print(scaled_A)


dim = scaled_A.shape[1]

turnOnHardware = 1
turnOnScaling = 0
#RESULT_MULT=1
RESULT_MULT= turnOnHardware+1
PRECISION=1e-6
RES_TOL= dim*dim*PRECISION*PRECISION
ITR_LIMIT=1000
ITR_TOL=dim*1e-2


rsmb = 1
pmb=1
ilmb=2

#best performance so far on device0
#numbitinput=10
#numbitpartialsum=10
# RES_TOL=1e-6
# PRECISION=1e-6
# ITR_LIMIT=20
#
# rsmb = 1
# pmb=1
# ilmb=2

INTERPOLATE=False
meliso_obj = meliso.MelisoPy(1,dim,dim,MAX_TOL,MIN_TOL,turnOnHardware,turnOnScaling)
# #Epiram
# meliso_obj.setConductanceProperties(1.2345679012345678e-05, 2.4592986080369874e-07,1.2345679012345678e-05, 2.4592986080369874e-07,2.4592986080369874e-07,2.4592986080369874e-07)
# meliso_obj.setWriteProperties(5.0, -3.0, 5e-6,5e-6,256,256)
# meliso_obj.setDeviceVariation(0.5,-0.5,0,0.02)

# if INTERPOLATE:
#     f,t = preprocess(meliso_obj,p,dim,dim,RESULT_MULT,RES_TOL,PRECISION,ITR_LIMIT)
#
# print(f,t)

#obtain an A matrix with values between 0,1
#I have observed that having matrix between 0,1 gives the best results


x_raw = np.loadtxt(fname='input_x',delimiter=',')
x = x_raw.reshape(x_raw.shape[0],1)[:dim]
x_ref= np.copy(x)

real_Ax = np.dot(scaled_A,x).reshape((1,dim)).flatten()

meliso_obj.initializeWeights()
scaled_A_j,scaled_A_res = setWeightsIncremental(meliso_obj,scaled_A,rsmb*RES_TOL,pmb*PRECISION,ilmb*ITR_LIMIT)
meliso_obj.loadInput(x)
meliso_obj.matVec()
y_benchmark_mem_result = RESULT_MULT*meliso_obj.getResults()

y_benchmark_mem_result = y_benchmark_mem_result.reshape((1,dim)).flatten()
benchmark_norm = np.linalg.norm(real_Ax-y_benchmark_mem_result,ord=np.inf)

meliso_obj.initializeWeights()
#y_rescaled_mem_result,X_j,A_j,X_res,A_res = correction(meliso_obj, scaled_A, x, RESULT_MULT,RES_TOL,PRECISION,ITR_LIMIT)
y_rescaled_mem_result,X_j,A_j,X_res,A_res = correctionADMMDenoising(meliso_obj, scaled_A, x, RESULT_MULT,RES_TOL,PRECISION,ITR_LIMIT,ITR_TOL,y_benchmark_mem_result,benchmark_norm)

y_rescaled_mem_result = y_rescaled_mem_result.reshape((1,dim)).flatten()
if INTERPOLATE:
    y0_rescaled_mem_result, X0_j, A0_j,X0_res,A0_res = correction(meliso_obj, scaled_A, 0 * x, RESULT_MULT, RES_TOL, PRECISION,
                                                    ITR_LIMIT)
    f, t = preprocess(meliso_obj, p, dim, dim, RESULT_MULT, RES_TOL, PRECISION, ITR_LIMIT)
    print(f,t)
    y0 = y0_rescaled_mem_result.flatten()
    cols = dim
    rows=dim
    for i in range(dim):
        xsum = np.sum(x_ref)

        ot = np.zeros((1, cols))
        Ot = np.tile(ot, (rows, 1))

        Ot_j, Ot_res = setWeightsIncremental(meliso_obj, Ot, 0.1, 0.1, 1)

        ones = np.ones(cols)
        meliso_obj.loadInput(ones)

        meliso_obj.matVec()

        Ot1=RESULT_MULT * meliso_obj.getResults()


        # res0 = evaluatePolynomial(p-1,Ot1[i],f[i,:],t[i,:])
        # res = evaluatePolynomial(p-1,xsum,f[i,:],t[i,:])

        res0 = evaluateLinearRegression(p - 1, Ot1[i], f[i, :], t[i, :])
        res = evaluateLinearRegression(p - 1, xsum, f[i, :], t[i, :])

        if Ot1[i][0] !=0.0:
            d0 = Ot1[i][0]
            dai = y0[i]/d0
            dxi = abs(res - xsum)
            dx = dxi #/dim
            corr = - dx*dai
            #corr = (dai*dxi)/float(dim)
            print("i:{},xsum:{},dai:{},dxi:{},corr:{},Ot1:{},y0:{},res:{},res0:{}".format(i,xsum,dai,dxi,corr,Ot1[i][0],y0[i], res, res0))
            #y_rescaled_mem_result[0,i] = y_rescaled_mem_result[0,i] + corr

# print(y0_rescaled_mem_result.flatten())
# print(y_rescaled_mem_result.flatten())



print("y_benchmark:",y_benchmark_mem_result.reshape((1,dim)))
print("y_correction:",y_rescaled_mem_result.reshape((1,dim)))
print("real_Ax:",real_Ax.reshape((1,dim)))

correction_norm = np.linalg.norm(real_Ax-y_rescaled_mem_result,ord=np.inf)
print("benchmark_norm",benchmark_norm)
print("correction_norm",correction_norm)
print(real_Ax-y_rescaled_mem_result)
print(real_Ax-y_benchmark_mem_result)
print("X_j:{},A_j:{},X_j+A_j:{},scaled_A_j:{}".format(X_j,A_j,X_j[0]+A_j,scaled_A_j))
print("X_res:{},A_res:{},scaled_A_res:{}".format(X_res,A_res,scaled_A_res))
# cols=rows=dim
# xt = x.reshape((1, cols))
# scaled_A = np.tile(xt,(rows,1))

#meliso_obj.initializeWeights()

# DA_tilde = A_tilde - A
#
#     DX_tilde = X_tilde - X
#
#     DAX_tilde = np.empty((rows, 1), dtype=float)
#
#     DXA_tilde = np.empty((rows, 1), dtype=float)
#
#     DAX_j = 0
#     DXA_j = 0

    # DAX_j,DAX_res = setWeightsIncremental(meliso_obj, DA_tilde, res_tol, precision, 0.2*itr_limit)
    #
    #
    #
    # for i in range(rows):
    #     dxit = DX_tilde[i,:].flatten()
    #     dax_tilde = matVec(meliso_obj,dxit,RESULT_MULT)
    #     DAX_tilde[i] = dax_tilde[i]
    #
    # DXA_j, DXA_res = setWeightsIncremental(meliso_obj, DX_tilde, res_tol, precision, 0.2*itr_limit)
    #
    #
    #
    # for i in range(rows):
    #     dait = DA_tilde[i,:].flatten()
    #     dxa_tilde = matVec(meliso_obj,dait,RESULT_MULT)
    #     DXA_tilde[i] = dxa_tilde[i]