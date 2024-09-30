import meliso, time, os
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

start_time = time.time()

correction=False
mv = MatVecSolver()
mv.matVec(correction=correction)
mv.finalize()
mv.acquireMCAStats()

if int(os.environ["CENTRALIZED"]) == 1:
    mv.centralizedBenchmarkMatVec()
    mv.finalize()
    end_time = time.time()
    print(f"Elapsed time for the entire VMM operation: {end_time - start_time}")
    exit()
else:
    mv.parallelizedBenchmarkMatVec(0,0,correction=correction)
    mv.finalize()
    end_time = time.time()
    print(f"Elapsed time for the entire VMM operation: {end_time - start_time}")
    exit()
