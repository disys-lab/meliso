import meliso
import numpy as np
# from src.core.Matvec import Matvec
# mv = MatVec()
#
# mv.parallelMatVec()
#
# mv.benchmarkMatVecParallel(0,0)
#

from solver.matvec.MatVecSolver import MatVecSolver

mv = MatVecSolver()

mv.matVec()

mv.finalize()

mv.parallelizedBenchmarkMatVec(0,0)

mv.finalize()