from mpi4py import MPI
import numpy as np
import os,sys

from Root import rootSolve
from NonRoot import nonRootSolve

if MPI.COMM_WORLD.Get_rank() == MPI.COMM_WORLD.Get_size() - 1:
    rootSolve()
else:
    nonRootSolve()
