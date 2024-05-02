from solver.matvec.MatVecSolver import MatVecSolver
def nonRootSolve():
    mv = MatVecSolver()
    mv.solverObject.parallelMatVec()
    mv.finalize()
