import meliso, time, os, sys
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver

REPORT_PATH = os.environ["REPORT_PATH"]

# 
start_time = time.time()
correction = False
mv = MatVecSolver()
mv.matVec(correction=correction)
mv.finalize()
mv.acquireMCAStats()
mv.parallelizedBenchmarkMatVec(0,0,correction=correction)
mv.finalize()
end_time = time.time()
#

duration = end_time - start_time
print(f"Elapsed time for the entire VMM operation: {duration}\n")

with open(REPORT_PATH, "a+") as file:
    file.write(f"Elapsed time for the entire VMM operation: {duration}\n")