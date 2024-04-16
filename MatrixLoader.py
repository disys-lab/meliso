import numpy as np
import scipy as sp

def load_matrix_object_from_MATLAB(filepath):
  data = sp.io.loadmat(filepath)
  matrix = data['Problem']['A'][0,0]
  matrix = matrix.toarray()
  return matrix

def read_matrix(filepath, format):
  if format == "mat":
    return load_matrix_object_from_MATLAB(filepath)

def check_matrix(matrix):
    matrix_norm	= np.linalg.norm(matrix, ord=2)
    print(f"Matrix Norm: {matrix_norm}")

    _, S, _ = np.linalg.svd(matrix); min_svd_value = min(S.reshape(-1))
    print(f"Minimum Singular Value: {min_svd_value}")

    condition_number = np.linalg.cond(matrix)
    print(f"Condition Number: {condition_number}")

    rank = S.shape[0]
    print(f"Rank: {rank}")