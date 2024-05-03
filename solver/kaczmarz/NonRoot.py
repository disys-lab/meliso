from solver.matvec.MatVecSolver import MatVecSolver
def nonRootSolve():
    mv = MatVecSolver()
    mv.parallelizedBenchmarkMatVec(0, 0)
    mv.finalize()
    mv.solverObject.parallelMatVec()
    mv.finalize()
