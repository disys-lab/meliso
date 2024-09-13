import meliso, time
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

correction=False

start_time = time.time()

mv = MatVecSolver()
mv.matVec(correction=correction)
mv.finalize()
mv.acquireMCAStats()
mv.parallelizedBenchmarkMatVec(0,0,correction=correction)
mv.finalize()

end_time = time.time()
print(f"Elapsed time for the entire VMM operation: {end_time - start_time}")