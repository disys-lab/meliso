import os
import numpy as np

class MLP:
    def __init__(self, W1_path, B1_path, W2_path, B2_path):
        self.W1, self.B1, self.W2, self.B2 = self.__load_model__(W1_path, B1_path, W2_path, B2_path)
    
    def predict(self, input_vector):
            """Perform inference with the MLP model given an input vector."""
            # 1. Convert to numpy array just in case a list is passed
            input_vector = np.asarray(input_vector)
            
            # 2. If the input is multi-dimensional (e.g., 28x28), flatten it to 1D
            if input_vector.ndim > 1:
                input_vector = input_vector.flatten()
                
            # 3. Sanity check: Ensure the vector length matches W1's expected input (784)
            expected_size = self.W1.shape[1]
            if input_vector.size != expected_size:
                raise ValueError(f"Input size mismatch. Expected {expected_size} elements, but got {input_vector.size}.")

            # First layer
            z1 = np.dot(self.W1, input_vector) + self.B1
            a1 = self.__relu__(z1)

            # Second layer
            z2 = np.dot(self.W2, a1) + self.B2
            a2 = self.__softmax__(z2)
            
            return z1, z2, a1,a2
    
    #-----------------------------------------------------------------------------------------------
    # Internal methods for the MLP class
    #-----------------------------------------------------------------------------------------------
    def __relu__(self, x):
        """ReLU activation function."""
        return np.maximum(0, x)
    
    def __softmax__(self, x):
        """Numerically stable softmax function."""
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum(axis=0)
    
    def __load_model__(self, W1_path, B1_path, W2_path, B2_path):
        """Load the two-layer MLP model parameters from the specified file paths."""
        W1 = np.load(W1_path, allow_pickle=True)
        B1 = np.load(B1_path, allow_pickle=True)
        W2 = np.load(W2_path, allow_pickle=True)
        B2 = np.load(B2_path, allow_pickle=True)
        print("Loaded MLP model parameters:"
              f"\nW1 shape: {W1.shape}, B1 shape: {B1.shape}"
              f"\nW2 shape: {W2.shape}, B2 shape: {B2.shape}")
        return W1, B1, W2, B2