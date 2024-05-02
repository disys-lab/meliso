from solver.matvec.MatVecSolver import MatVecSolver

def rootSolve():
    mv = MatVecSolver()

    for k in range(10):

        #implement row selection
        i = np.random.randint(0,mv.mca.maxVRows)
        for j in range(mv.mca.maxVRows):
            mv.mca.virtualParallelMatVec( i, j)

            #obtain y
            y = mv.y_mem_result

            #use to implement other aspects of Kaczmarz

            #a new A matrix can be re-initialized using:
            #mv.solverObject.initializeMatrix(new_mat)

            #initialize a new x using:
            #mv.solverObject.initializeX(new_x)

            #Remember!: always initialize new matrix before initializing a new vector (if your new matrix has different dimensions)



        mv.matVec()

    mv.finalize()
