#===================================================================================================
# DRIVER SCRIPT FOR MLP INFERENCE WITH ACCELERATED MVM USING MELISO+
#===================================================================================================
"""
@author: Huynh Quang Nguyen Vo
@affiliation: Oklahoma State University
This script demonstrates how to perform inference with a simple multi-layer perceptron (MLP) using 
the MELISO+ framework for accelerated matrix-vector multiplication (MVM). It loads a pre-trained MLP 
model, runs inference on the MNIST test set, and evaluates the accuracy of the predictions.
"""

import os
import meliso
import subprocess
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver
from solver.mlp.MLP import MLP

# --------------------------------------------------------------------------------------------------
# Utility functions
# --------------------------------------------------------------------------------------------------
def cancel_SLURM_job():
    job_id = os.environ.get('SLURM_JOB_ID')
    if job_id:
        print(f"Cancelling job: {job_id}")
        # Call the scancel command
        subprocess.run(['scancel', job_id])
    else:
        print("Not running in a SLURM environment.")

#---------------------------------------------------------------------------------------------------
# Main function to run MLP inference and evaluate accuracy
#---------------------------------------------------------------------------------------------------
def main():
    model = MLP(W1_path="./inputs/mlp/W1.npy", 
                B1_path="./inputs/mlp/B1.npy", 
                W2_path="./inputs/mlp/W2.npy", 
                B2_path="./inputs/mlp/B2.npy")
    
    # Load the MNIST test set (10,000 samples)
    test_images = np.load("./inputs/mlp/mnist_test_images.npy") # Shape: (10000, 784)
    test_labels = np.load("./inputs/mlp/mnist_test_labels.npy") # Shape: (10000,)

    # ---------------------------------------------------------------------------------------------- 
    # Run inference on a subset of the test set and evaluate accuracy on CPU
    # ----------------------------------------------------------------------------------------------
    correct_predictions = 0
    subset_size = 1 # Adjust this to test on more or fewer samples 
    for i in range(len(test_images)):
        if i >= subset_size:
            break

        input_vector = test_images[i] # Shape: (784,)
        true_label = test_labels[i]
        z1_cpu, z2_cpu, a1_cpu, a2_cpu  = model.predict(input_vector)
        predicted_label = np.argmax(a2_cpu)
        print(f"Sample {i}: True label = {true_label}, Predicted label = {predicted_label}")
        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / subset_size
    print(f"Accuracy on the first {subset_size} samples of the MNIST test set: {accuracy:.2%}")

    # ----------------------------------------------------------------------------------------------
    # Run inference on the same subset using MELISO+ for accelerated MVM
    # ----------------------------------------------------------------------------------------------
    correct_predictions_meliso = 0
    for i in range(len(test_images)):
        if i >= subset_size:
            break

        input_vector = test_images[i] # Shape: (784,)
        input_vector = input_vector.reshape(-1,1) # Reshape to (784, 1) for MVM
        true_label = test_labels[i]

        # Setup MELISO+ for the first time
        CORRECTION = True # With min-max scaling reversion 
        mv = MatVecSolver(xvec=input_vector, mat=model.W1) # Initialize with the first layer weights

        # --- First layer, (784, 1) x (512, 784) + (512,1) -> (512, 1) ---
        z1, z2, a1, a2 = np.zeros((512,1)), np.zeros((10,1)), np.zeros((512,1)), np.zeros((10,1)) 

        # 1. Distributed matrix-vector multiplication (MVM) on MELISO+
        mv.matVec(correction=CORRECTION) 
        mv.finalize() # Memristive MVM should be in the original range
        mv.acquireMCAStats() 
        z1 = np.loadtxt('./y_mem_result.txt', delimiter=',') # Load the MVM result for the first layer

        if z1 is not None and z1.min() != 0 and z1.max() != 0:
            print(f"Obtained MVM result with shape {z1.shape} for the first layer.")
            # 2. Add bias and apply ReLU activation
            z1 = z1[0]                  # Remove the extra dimension from MVM output
            print(f"Relative L2 error for the first layer MVM result compared to CPU: {np.linalg.norm(z1 - z1_cpu) / np.linalg.norm(z1_cpu):.2%}")
            z1 = z1 + model.B1          # Add bias
            a1 = model.__relu__(z1)     # Apply ReLU activation
            print(f"Relative L2 error for the first layer activated output compared to CPU: {np.linalg.norm(a1 - a1_cpu) / np.linalg.norm(a1_cpu):.2%}")
            print(f"Obtained activated output with shape {a1.shape} for the first layer.")
        
        # --- Second layer, (512, 1) x (10, 512) + (10,1) -> (10, 1) ---
        # TODO: For the second layer, we would need to re-initialize the MatVecSolver with the 
        # new input vector (a1) and the second layer weights (model.W2). A more elegent approach is 
        # needed.
        mv = MatVecSolver(xvec=a1.reshape(-1,1), mat=model.W2) # Initialize with the second layer weights
        mv.matVec(correction=CORRECTION)
        mv.finalize() # Memristive MVM should be in the original range
        mv.acquireMCAStats()
        z2 = np.loadtxt('./y_mem_result.txt', delimiter=',') # Load the MVM result for the second layer
        mv.stopCommunication() # Cleanly stop MPI communication after acquiring results

        if z2 is not None and z2.min() != 0 and z2.max() != 0:
            print(f"Obtained MVM result with shape {z2.shape} for the second layer.")
            # 3. Add bias and apply softmax activation
            z2 = z2[0]                  # Remove the extra dimension from MVM output
            print(f"Relative L2 error for the second layer MVM result compared to CPU: {np.linalg.norm(z2 - z2_cpu) / np.linalg.norm(z2_cpu):.2%}")
            z2 = z2 + model.B2          # Add bias
            a2 = model.__softmax__(z2)  # Apply softmax activation
            print(f"Relative L2 error for the second layer activated output compared to CPU: {np.linalg.norm(a2 - a2_cpu) / np.linalg.norm(a2_cpu):.2%}")
            print(f"Obtained activated output with shape {a2.shape} for the second layer.")

            predicted_label_meliso = np.argmax(a2)
            print(f"Sample {i}: True label = {true_label}, Predicted label (MELISO+) = {predicted_label_meliso}")
            if predicted_label_meliso == true_label:
                correct_predictions_meliso += 1
            
    accuracy_meliso = correct_predictions_meliso / subset_size
    print(f"Accuracy on the first {subset_size} samples of the MNIST test set using MELISO+: {accuracy_meliso:.2%}")

if __name__ == "__main__":
    main()