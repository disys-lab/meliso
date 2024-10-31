import os
import re
import csv
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import os

# Folder containing Matrix Market files
folder_path = './inputs/matrices/'

# List to store each matrix's properties
matrix_properties = []

# Function to calculate properties of a matrix and store them in the list
def analyze_matrix(matrix_path):
    matrix = scipy.io.mmread(matrix_path).tocsc()
    
    # Properties of the matrix
    nrows, ncols = matrix.shape
    condition_number = np.linalg.cond(matrix.toarray()) if nrows == ncols else None
    total_elements = nrows * ncols
    non_zero_elements = matrix.nnz
    sparsity = (1 - non_zero_elements / total_elements) * 100
    is_symmetric = (matrix != matrix.T).nnz == 0
    rank = np.linalg.matrix_rank(matrix.toarray())
    
    # Add properties to list
    matrix_properties.append({
        'Matrix Name': os.path.basename(matrix_path),
        'Rows': nrows,
        'Columns': ncols,
        'Condition Number': condition_number,
        'Sparsity (%)': sparsity,
        'Symmetric': is_symmetric,
        'Rank': rank
    })

# Get all .mtx files in the folder
matrix_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path) if f.endswith('.mtx')]

# Iterate over each matrix file
for matrix_file in matrix_files:
    analyze_matrix(matrix_file)

# Create a DataFrame from the collected properties
df = pd.DataFrame(matrix_properties)

# Display or save the DataFrame as needed
print(df)
