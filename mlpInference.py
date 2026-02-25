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
import numpy as np
from solver.matvec.MatVecSolver import MatVecSolver
from solver.mlp.MLP import MLP

# --------------------------------------------------------------------------------------------------
# Distributed matrix-vector multiplication (MVM) wrapper with min-max scaling reversion
# --------------------------------------------------------------------------------------------------
def melisoMVM(input_vector, input_matrix, verbose=False):
    """
    A wrapper function to perform distributed matrix-vector multiplication (MVM) using MELISO+ with
    min-max scaling reversion.
    In memristive devices, the matrices containing positive and negative weights are scaled to fit 
    within the device's conductance range. Here, 
    1. We apply min-max scaling to scale the weights to the [0,1] range for MVM. 
    2. After obtaining the MVM results, we apply min-max scaling reversion to recover the original 
       range of the outputs.
    """
    correction = True # With min-max scaling reversion
    mv = MatVecSolver(xvec=input_vector, mat=input_matrix)
    mv.matVec(correction=correction)
    mv.finalize() # Memristive MVM should be in the original range
    mv.acquireMCAStats()
    output_vector = mv.acquireResults()
    if verbose:
        print("Obtained MVM result with min-max scaling reversion: \n", output_vector)
    return output_vector

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
        predicted_label = np.argmax(model.predict(input_vector))

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

        # First layer MVM using MELISO+
        z1 = melisoMVM(input_vector, model.W1) + model.B1.reshape(-1,1) # Shape: (128, 1)
        a1 = model.__relu__(z1)

        # Second layer MVM using MELISO+
        z2 = melisoMVM(a1, model.W2) + model.B2.reshape(-1,1) # Shape: (10, 1)
        a2 = model.__softmax__(z2)

        predicted_label = np.argmax(a2)
        if predicted_label == true_label:
            correct_predictions_meliso += 1
    accuracy_meliso = correct_predictions_meliso / subset_size
    print(f"Accuracy on the first {subset_size} samples of the MNIST test set using MELISO+: {accuracy_meliso:.2%}")

if __name__ == "__main__":
    main()