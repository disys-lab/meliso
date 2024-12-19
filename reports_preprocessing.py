import os
import re
import csv
import pandas as pd
import numpy as np
import scipy.io
import scipy.sparse
import os

def extract_data(top_dir, output_csv_filename, include_min_elapsed_time=False):
    # Initialize an empty list to store the extracted data
    data_list = []

    # Walk through the directory structure
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
                if include_min_elapsed_time:
                    elapsed_times = []
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
                        # Extract Elapsed time and collect all instances
                        if include_min_elapsed_time:
                            match_elapsed = re.search(r'Elapsed time for the entire VMM operation: ([\d\.eE+-]+)', line)
                            if match_elapsed:
                                elapsed_time = float(match_elapsed.group(1))
                                elapsed_times.append(elapsed_time)
                # Determine the smallest elapsed time
                if include_min_elapsed_time:
                    if elapsed_times:
                        min_elapsed_time = min(elapsed_times)
                    else:
                        min_elapsed_time = None
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
                if include_min_elapsed_time:
                    data['min_elapsed_time'] = min_elapsed_time
                data_list.append(data)

    # Define the CSV file headers
    fieldnames = [
        'material', 'exp_id', 'iteration', 'replication',
        'writeLatency_mean', 'writeEnergy_mean',
        'Loo_norm_Error', 'L2_norm_Error'
    ]
    if include_min_elapsed_time:
        fieldnames.append('min_elapsed_time')

    # Write the extracted data to a CSV file
    with open(output_csv_filename, 'w+', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for data in data_list:
            writer.writerow(data)

    return None
    
