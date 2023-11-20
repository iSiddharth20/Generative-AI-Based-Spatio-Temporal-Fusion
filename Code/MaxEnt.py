'''
Module that specifies Maximum Entropy Modelling (MaxEnt)
--------------------------------------------------------------------------------
Data Formats Used:
    - low_res_image: [M x N] NumPy Array
    - high_res_image: [M' x N'] NumPy Array
    - estimated_high_res_image: [M' x N'] NumPy Array
The 'L-BFGS-B' method is used because it's good for large-scale optimization problems
--------------------------------------------------------------------------------
This Code Needs to be Modified based on the Problem Statement 
'''


# Defining the MaxEnt Model
import numpy as np
from scipy.optimize import minimize

class MaxEnt(): 
    def __init__(self, low_res_image, high_res_image):
        self.low_res_image = low_res_image
        self.high_res_image = high_res_image

    # Calculate difference between the High-Resolution Image and Estimated/Predicted High-Resolution Image
    def objective_function(self, estimated_high_res_image):
        return np.sum((self.high_res_image - estimated_high_res_image)**2)

    # Use the MaxEnt Principle to Create a High-Resolution Version of the Low-Resolution Image
    def MaxEnt(self):
        result = minimize(self.objective_function, self.low_res_image, method='L-BFGS-B')
        return result.x