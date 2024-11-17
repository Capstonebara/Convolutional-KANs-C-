cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(CheckLibTorch)

# Ensure CUDA is not being enabled
set(CMAKE_CUDA_COMPILER "" CACHE STRING "CUDA compiler" FORCE)

# Set path to the libtorch library
set(Torch_DIR "/home/nhomnhom0/miniforge3/envs/Conv-KAN-env/lib/python3.9/site-packages/torch/share/cmake/Torch")

# Find the PyTorch package
find_package(Torch REQUIRED)

# Include the layers directory to locate headers
include_directories(${CMAKE_SOURCE_DIR}/include)
include_directories(${CMAKE_SOURCE_DIR}/layers)

# Add the executable
add_executable(test src/layers/KAN_linear.cpp tests/test_reset_parameters.cpp)

# Link to PyTorch (CPU-only version)
target_link_libraries(test "${TORCH_LIBRARIES}")

# Set C++ standard to 17 (required by PyTorch)
set_property(TARGET test PROPERTY CXX_STANDARD 17)
