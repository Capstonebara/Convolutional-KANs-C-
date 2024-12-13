import torch
import numpy as np
import sys
import os
# Dynamically add the project root directory to sys.path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))
sys.path.append('/home/nhomnhom0/Convolutional-KANs-C/tests')

from test_conv_and_KAN.conv_and_kan import NormalConvsKAN_Medium

# Test Case 1: Test with a single input image (batch size = 1)
def test_single_input():
    # Create a simple input tensor of shape (batch_size=1, channels=1, height=5, width=5)
    input_tensor = torch.ones(1, 1, 28, 28)
    
    # Initialize the model
    model = NormalConvsKAN_Medium(grid_size=5)
    
    # Run the model
    output = model(input_tensor)
    
    # Check the output shape (should be [batch_size=1, num_classes=10])
    print("Test 1 Output Shape:", output.shape)
    print("Test 1 Output:", output)

# # Test Case 2: Test with a batch of 3 images
# def test_batch_input():
#     # Create a batch of 3 images with size (batch_size=3, channels=1, height=5, width=5)
#     input_tensor = torch.tensor([[[[1.0, 0.0, 0.0, 0.0, 1.0], 
#                                   [1.0, 0.0, 0.0, 0.0, 1.0],
#                                   [1.0, 0.0, 0.0, 0.0, 1.0],
#                                   [1.0, 0.0, 0.0, 0.0, 1.0],
#                                   [1.0, 0.0, 0.0, 0.0, 1.0]]], 
#                                   [[[0.0, 1.0, 1.0, 1.0, 0.0], 
#                                     [0.0, 1.0, 1.0, 1.0, 0.0],
#                                     [0.0, 1.0, 1.0, 1.0, 0.0],
#                                     [0.0, 1.0, 1.0, 1.0, 0.0],
#                                     [0.0, 1.0, 1.0, 1.0, 0.0]]], 
#                                   [[[0.5, 0.5, 0.5, 0.5, 0.5], 
#                                     [0.5, 0.5, 0.5, 0.5, 0.5],
#                                     [0.5, 0.5, 0.5, 0.5, 0.5],
#                                     [0.5, 0.5, 0.5, 0.5, 0.5],
#                                     [0.5, 0.5, 0.5, 0.5, 0.5]]]], dtype=torch.float32)  # (3, 1, 5, 5)

#     # Initialize the model
#     model = NormalConvsKAN_Medium(grid_size=5)
    
#     # Run the model
#     output = model(input_tensor)
    
#     # Check the output shape (should be [batch_size=3, num_classes=10])
#     print("Test 2 Output Shape:", output.shape)
#     print("Test 2 Output:", output)

# Run the test cases
test_single_input()
# test_batch_input()