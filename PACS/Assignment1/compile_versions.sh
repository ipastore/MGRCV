#!/bin/bash

# Create the build directory if it doesn't exist
mkdir -p build

# Directories for Eigen (if needed)
eigen_dir="./eigen-3.4.0"

# Compile the standard matrix multiplication versions
g++ standard_matrix.cpp -o build/standard_matrix
g++ -O2 standard_matrix.cpp -o build/standard_matrix_O2
g++ -O3 standard_matrix.cpp -o build/standard_matrix_O3

# Compile the Eigen-based versions
g++ -I $eigen_dir my_eigen_matmult.cpp -o build/my_eigen_matmult
g++ -O2 -I $eigen_dir my_eigen_matmult.cpp -o build/my_eigen_matmult_O2
g++ -O3 -I $eigen_dir my_eigen_matmult.cpp -o build/my_eigen_matmult_O3

# Compile static memory version
g++ static_standard_matrix.cpp -o build/static_standard_matrix

# Compile multi-threading versions
g++ -fopenmp standard_matrix.cpp -o build/standard_matrix_threading
g++ -fopenmp -I $eigen_dir my_eigen_matmult.cpp -o build/my_eigen_matmult_threading

echo "All programs compiled and placed in 'build' folder."
