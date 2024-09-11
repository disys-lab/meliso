import meliso
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

correction=False

mv = MatVecSolver()

mv.matVec(correction=correction)
mv.finalize()
mv.parallelizedBenchmarkMatVec(0,0,correction=correction)
# mv.acquireMCAStats()
mv.finalize()