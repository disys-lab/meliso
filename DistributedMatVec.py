import meliso
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

correction=True
mv = MatVecSolver()
mv.matVec(correction=correction)
mv.finalize()
mv.acquireMCAStats()

mv.parallelizedBenchmarkMatVec(0,0,correction=correction)
mv.finalize()