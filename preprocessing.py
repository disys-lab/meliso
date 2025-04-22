import os
import re
import csv
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
from typing import List, Dict, Any

def extract_data(top_dir: str, output_csv_filename: str) -> None:
    """
    Extracts data from text files in a directory structure and saves them to a CSV file.

    1. The directory structure is expected to be:
        'top_dir/material_name/exp_id_iter_rep.txt
        where:
        - material_name is the name of the material (directory name)
        - exp_id is the experiment ID (part of the filename)
        - iter is the iteration number (part of the filename)
        - rep is the replication number (part of the filename)

    2. The text files contain lines with the following patterns:
        - "writeLatency mean = <value>"
        - "writeEnergy mean = <value>"
        - "Relative Loo-norm Error: <value>"
        - "Relative L2-norm Error: <value>"
        - "Elapsed time for the entire VMM operation: <value>"

    3. The extracted data is saved to a CSV file with the following columns:
        - material
        - exp_id
        - iteration
        - replication
        - writeLatency_mean
        - writeEnergy_mean
        - Loo_norm_Error
        - L2_norm_Error

    @params:
        top_dir (str): The top-level directory containing the material directories.
        output_csv_filename (str): The name of the output CSV file.
    @returns:
        None   
    """
    # --- Initialize an empty list to store the extracted data ---
    data_list = []

    # --- Walk through the directory structure ---
    for root, dirs, files in os.walk(top_dir):
        for file in files:
            if file.endswith('.txt'):
                file_path = os.path.join(root, file)
                # Extract the material name from the directory name
                material_name = os.path.basename(root)
                # Extract experiment ID, iteration number, and replication number from the filename
                match = re.match(r'(exp\d+)_iter_(\d+)_rep_(\d+)\.txt', file)
                if match:
                    exp_id = match.group(1)
                    iter_num = int(match.group(2))
                    rep_num = int(match.group(3))
                else:
                    exp_id = ''
                    iter_num = ''
                    rep_num = ''
                # Initialize variables to None
                writeLatency_mean = None
                writeEnergy_mean = None
                Loo_norm_Error = None
                L2_norm_Error = None

                # Open and read each file
                with open(file_path, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        line = line.strip()
                        # Extract writeLatency mean
                        match_latency = re.search(r'writeLatency mean = ([^,]+)', line)
                        if match_latency:
                            writeLatency_mean = float(match_latency.group(1))
                        # Extract writeEnergy mean
                        match_energy = re.search(r'writeEnergy mean = ([^,]+)', line)
                        if match_energy:
                            writeEnergy_mean = float(match_energy.group(1))
                        # Extract Loo-norm Error
                        match_loo_error = re.search(r'Relative Loo-norm Error: ([\d\.eE+-]+)', line)
                        if match_loo_error:
                            Loo_norm_Error = float(match_loo_error.group(1))
                        # Extract L2-norm Error
                        match_l2_error = re.search(r'Relative L2-norm Error: ([\d\.eE+-]+)', line)
                        if match_l2_error:
                            L2_norm_Error = float(match_l2_error.group(1))

                # Compile the extracted data into a dictionary
                data = {
                    'material': material_name,
                    'exp_id': exp_id,
                    'iteration': iter_num,
                    'replication': rep_num,
                    'writeLatency_mean': writeLatency_mean,
                    'writeEnergy_mean': writeEnergy_mean,
                    'Loo_norm_Error': Loo_norm_Error,
                    'L2_norm_Error': L2_norm_Error,
                }
                data_list.append(data)

    # --- Define the CSV file headers ---
    fieldnames = [
        'material', 'exp_id', 'iteration', 'replication',
        'writeLatency_mean', 'writeEnergy_mean',
        'Loo_norm_Error', 'L2_norm_Error'
    ]

    # --- Write the extracted data to a CSV file ---
    with open(output_csv_filename, 'w+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)

def preprocess_data(data: pd.DataFrame, save_path: str, experiment_ids: List, 
                    iterations: List, materials: List, quantities: List):
    """
    Preprocesses the data by filtering and grouping it, then saves it to a CSV file.
    """
    filtered_data = data[
    (data['iteration'].isin(iterations)) & 
    (data['material'].isin(materials)) & 
    (data['exp_id'].isin(experiment_ids))
    ]
    
    # --- Group data by material, experiment, and iteration ---
    grouped_data = filtered_data.groupby(['material', 'exp_id', 'iteration'])\
        [quantities].mean().reset_index()

    # --- Save the processed data to an Excel file ---
    grouped_data.to_csv(save_path, index=False, header=True)
    return None

if __name__ == "__main__":
    extract_data("./reports/iterations", "varied_iterations.csv")