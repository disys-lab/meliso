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

correction=True

mv = MatVecSolver()

mv.matVec(correction=correction)

mv.finalize()

mv.parallelizedBenchmarkMatVec(0,0,correction=correction)

mv.finalize()