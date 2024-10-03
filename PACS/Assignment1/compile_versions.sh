#!/bin/bash

# Create the build directory if it doesn't exist
mkdir -p build

# #Standard version in stack (array[N][N])*
g++ ./src/standard_matrix_stack.cpp -o ./build/standard_matrix_stack
g++ -O2 ./src/standard_matrix_stack.cpp -o ./build/standard_matrix_stack_O2
g++ -O3 ./src/standard_matrix_stack.cpp -o ./build/standard_matrix_stack_O3

# # Standard version in stack with flattened vector  (array[N*N])
g++ ./src/standard_matrix_stack_flattened_array.cpp -o ./build/standard_matrix_stack_flattened_array
g++ -O2 ./src/standard_matrix_stack_flattened_array.cpp -o ./build/standard_matrix_stack_flattened_array_O2
g++ -O3 ./src/standard_matrix_stack_flattened_array.cpp -o ./build/standard_matrix_stack_flattened_array_O3

# Standard version in heap
g++ ./src/standard_matrix_heap.cpp -o ./build/standard_matrix_heap
g++ -O2 ./src/standard_matrix_heap.cpp -o ./build/standard_matrix_heap_O2
g++ -O3 ./src/standard_matrix_heap.cpp -o ./build/standard_matrix_heap_O3

# Standard version in heap with flattened vector
g++ ./src/standard_matrix_heap_flattened_version.cpp -o ./build/standard_matrix_heap_flattened_version
g++ -O2 ./src/standard_matrix_heap_flattened_version.cpp -o ./build/standard_matrix_heap_flattened_version_O2
g++ -O3 ./src/standard_matrix_heap_flattened_version.cpp -o ./build/standard_matrix_heap_flattened_version_O3

# Using EIGEN library
g++ -I ./eigen-3.4.0 ./src/my_eigen_matmult.cpp -o ./build/my_eigen_matmult
g++ -O2 -I ./eigen-3.4.0 ./src/my_eigen_matmult.cpp -o ./build/my_eigen_matmult_O2
g++ -O3 -I ./eigen-3.4.0 ./src/my_eigen_matmult.cpp -o ./build/my_eigen_matmult_O3

echo "All programs compiled and placed in 'build' folder."
